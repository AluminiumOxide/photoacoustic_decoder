import copy
import os.path
from PIL import Image
import torch
import numpy as np

from torch.autograd import Variable
from torch.utils.tensorboard import SummaryWriter

from reconstruct.Loss import mix_loss, TVLoss
from reconstruct.post_process import draw_img,draw_img_p0

from reconstruct.mcx_try_function import *


def load_net_input(opt, input_image):
    if opt.net_input is not None:
        print("input provided")
    else:
        # feed uniform noise into the network
        totalupsample = 2 ** len(opt.num_channels)  # totalupsample=2^4
        width = int(input_image.data.shape[2] / totalupsample)  # 图片的长/上采样次数得到初始随机张量的尺寸
        height = int(input_image.data.shape[3] / totalupsample)
        shape = [1, opt.num_channels[0], width, height]  # shape=[1,64,16,16]

        net_const = torch.zeros(shape).requires_grad_(True)  # net_const 随机的储存大小确定并且全部给0
        torch.manual_seed(429)
        net_const.data.uniform_()
        net_const.data *= 1. / 10
        print("InputNoise: with shape :{}  and example info :{}".format(net_const.shape, net_const[0][0][0][:8]) )

        return net_const


def load_optimizer(opt, model, net_input):

    p = [x for x in model.parameters()]  # 保存初始随机的网络自动生成的参数

    if opt.opt_input:  # optimizer over the input as well
        net_input.requires_grad = True
        p += [net_input]

    if opt.optimizer == 'SGD':
        print("optimize with SGD", opt.LR)
        optimizer = torch.optim.SGD(p, lr=opt.LR, momentum=0.9, weight_decay=opt.weight_decay)
    elif opt.optimizer == 'adam':
        print("optimize with adam", opt.LR)
        optimizer = torch.optim.Adam(p, lr=opt.LR, weight_decay=opt.weight_decay)  # 这里选择的是这个
    elif opt.optimizer == 'LBFGS':
        print("optimize with LBFGS", opt.LR)
        optimizer = torch.optim.LBFGS(p, lr=opt.LR)

    return optimizer


def load_loss(opt):
    mse = torch.nn.MSELoss().type(opt.dtype)
    tv = TVLoss().type(opt.dtype)
    save_loss_list = {
        'mse': [],
        'tv': [],
        'total': [],
    }
    return save_loss_list,mse,tv


def exp_lr_scheduler(optimizer, opt):
    """Decay learning rate by a factor of 0.1 every lr_decay_epoch epochs."""
    # decay_time = epoch//lr_decay_epoch
    # lr = lr * (0.1 ** decay_time)
    opt.LR = opt.LR * 0.6

    print('LR is set to {}'.format(opt.LR))
    for param_group in optimizer.param_groups:
        param_group['lr'] = opt.LR

    return optimizer,opt


def nowtime_mcxshape(epoch, total_epoch,x_shape,y_shape):
    """
    :param epoch: 入当前p0的迭代次数
    :param total_epoch: 总次数
    :param x_shape: 输出图像的x和y多大
    :param y_shape:
    :return: 返回目前mcx模拟的空间类似[1,x,y]或[x,y,z]
    """
    final_z = max([x_shape, y_shape])
    rate = 0.05

    # if epoch/total_epoch <= rate:
    #     mcxshape = [1, x_shape, y_shape]
    # else:  # 总感觉这里不太好，最好是找个s型曲线
    #     nowtime_step = (epoch-total_epoch*rate)/(total_epoch*(1-rate))  # 0~1
    #     z_shape = int(nowtime_step*nowtime_step*final_z)+3  # int(1/2)向下取整到0,但是我不喜欢用math的函数,就这样吧
    #     mcxshape = [x_shape, y_shape, 21]

    # mcxshape = [1, x_shape, y_shape]
    # mcxshape = [x_shape, y_shape,1]
    mcxshape = [x_shape, y_shape,50]
    return mcxshape


def nowtime_photons(epoch, total_epoch):
    begin_photon = 1e5
    end_photon = 1e8
    rate = 0.8
    now_rate = epoch/total_epoch
    if epoch/total_epoch <= rate:
        mcx_photons = begin_photon
    else:
        nowtime_step = (epoch - total_epoch * rate) / (total_epoch * (1 - rate))  # (1 - rate) 但我想给它拉平点，比如拽到1.5
        mcx_photons = nowtime_step*nowtime_step*(end_photon-begin_photon)+begin_photon
    return int(mcx_photons)


def nowtime_mcxinfo(epoch,total_epoch,x_shape,y_shape):
    """ 具体描述看上面的函数，这里计划返回一个包含 mcx_shape、mcx_photons的字典"""
    mcx_shape = nowtime_mcxshape(epoch, total_epoch, x_shape, y_shape)
    mcx_photons =nowtime_photons(epoch, total_epoch)
    mcx_info = {'mcx_shape':mcx_shape,
                'mcx_photons':mcx_photons,
                'epoch':epoch}
    return mcx_info


def evaluate_info(opt, model, label_img):
    """ label_img: tensor(1,1,256,256) """
    # opt.margin_flag = 0.1  # model的输出∈[0,1] 在该值上的全进行计算，该值以下全部置0  # 注意:该值与label边界判定的阈值并不是一个意思
    opt.ua_min = 0.0      # 预先设定,组织体吸收系数最小值
    opt.ua_max = 0.1       # 预先设定,组织体吸收系数最大值

    net_input_const = load_net_input(opt, label_img)
    optimizer = load_optimizer(opt, model, net_input_const)
    save_loss_list, mse, tv = load_loss(opt)

    if opt.find_best:
        best_net = copy.deepcopy(model)
        best_mse = 1000000

    margin_idx = torch.where(label_img < 0.00001)
    matrix_us = torch.full_like(label_img, 1)
    matrix_us[margin_idx] = 0

    writer = SummaryWriter(os.path.join(opt.save_path,'p0'))
    # writer.add_image('org_input', label_img[0].cpu().detach().numpy(), 1)

    mc_out = False
    for i in range(opt.num_iter_p0):  # --------------------------------------------------------------------------------
        # 学习率衰减
        if opt.lr_decay_epoch != 0 and i%opt.lr_decay_epoch==0:
            optimizer,opt = exp_lr_scheduler(optimizer, opt)
        # 原图添加随即抖动
        if opt.reg_noise_std > 0:
            if i % opt.reg_noise_decay == 0:
                opt.reg_noise_std *= 0.7
            net_input = net_input_const + torch.randn(net_input_const.shape) * opt.reg_noise_std

        # optimizer.zero_grad()
        # net_output = model(net_input.type(opt.dtype))
        # margin_label = net_output.clone()
        # margin_label[margin_idx] = 0
        # mse_loss_margin = mse(net_output, margin_label)
        # mse_loss_margin.backward()
        # optimizer.step()
        #
        optimizer.zero_grad()
        net_output = model(net_input.type(opt.dtype))

        """
        net_output作为一个1,1,256,256输入 (ua、us) [0,1]
        使用nowtime_mcxshape()根据当前迭代次数转换mcx的空间以及当前模拟光子数
        opt.ua_min 和 opt.ua_max 临时加的,后续将网络输出与该区间匹配
        using_mcx() 的输入分别是 opt, ua, us, margin, mcx_info, fai_tune_cache
        """
        mcxinfo = nowtime_mcxinfo(i, opt.num_iter_p0, x_shape=net_output.shape[2], y_shape=net_output.shape[3])

        mc_out, mc_mix = using_mcx(opt, net_output, matrix_us, margin_idx, mcxinfo,mc_out)  # 输出的分别是 fai 和 p0
        """ 调整后
        :param net_output: 网络输出 光子吸收系数分布 ua
        :param mc_out: 蒙卡输出 光通量分布 fai
        :param mc_mix: 蒙卡+网络计算得到的 初始声压图像 p0
        """
        # Huber_loss = nn.SmoothL1Loss(beta=1)
        mse_loss = mse(mc_mix, label_img)
        tv_loss = tv(mc_mix)
        total_loss = mse_loss
        # total_loss = mse_loss +  mse_loss_margin
        # --------------------------------------------------------------------------------------------------------------

        total_loss.backward()
        optimizer.step()

        if i % opt.print_step == 0:
            print('\nEvaluate_info: {} epoch {}'.format('p0', i))
            print('Evaluate_info: with mcx info {}'.format(mcxinfo))
            print('Evaluate_info: calcuate mse loss {} with {}'.format(net_output.shape, label_img.shape))
            print('Iteration %05d Total loss %f Mse loss %f TV loss %f ' % (
            i, total_loss.data, mse_loss.data, tv_loss.data, ), '\r')

            save_loss_list["mse"].append(mse_loss.data.cpu().numpy())
            save_loss_list["tv"].append(tv_loss.data.cpu().numpy())
            save_loss_list["total"].append(total_loss.data.cpu().numpy())

            if i % opt.draw_step == 0 or i <= opt.draw_step:  # 高频读写文件实在是太影响效率了
                out_draw = net_output.clone()
                # out_draw[margin_idx] = 0  # 强制掐边
                draw_img_p0(opt, label_img, net_input, mc_out ,mc_mix, out_draw, i)

            writer.add_image('output', net_output[0].cpu().detach().numpy(), i)
            # writer.add_image('output', net_output[0].cpu().detach().numpy(), i)
            writer.add_scalars('loss',{'Total loss': total_loss.data,
                                       'Mse loss': mse_loss.data,
                                       'TV loss': tv_loss.data},i)
            writer.add_scalars('learning_rate',{'lr':opt.LR},i)

        if opt.find_best:
            # if training loss improves by at least one percent, we found a new best net
            if best_mse > 1.005 * mse_loss.data:
                best_mse = mse_loss.data
                best_net = copy.deepcopy(model)

    writer.close()
    if opt.find_best:
        model = best_net

    net_out_p0 = model(net_input_const.type(opt.dtype)).data.cpu().numpy()[0][0]

    return save_loss_list, net_out_p0, model



# 当作ua的,以后再改
def pre_evaluate_info(opt, model, label_img):

    net_input_const = load_net_input(opt, label_img)
    optimizer = load_optimizer(opt, model, net_input_const)
    save_loss_list, mse, tv = load_loss(opt)

    if opt.find_best:
        best_net = copy.deepcopy(model)
        best_mse = 1000000

    writer = SummaryWriter(os.path.join(opt.save_path,'ua'))
    writer.add_image('org_input', label_img[0].cpu().detach().numpy(), 1)

    mc_out = False
    for i in range(opt.num_iter_ua):  # --------------------------------------------------------------------------------
        # 学习率衰减
        if opt.lr_decay_epoch != 0 and i % opt.lr_decay_epoch == 0:
            optimizer,opt = exp_lr_scheduler(optimizer, opt)
        # 原图添加随即抖动
        if opt.reg_noise_std > 0:
            if i % opt.reg_noise_decay == 0:
                opt.reg_noise_std *= 0.7
            net_input = net_input_const + torch.randn(net_input_const.shape) * opt.reg_noise_std

        optimizer.zero_grad()
        net_output = model(net_input.type(opt.dtype))

        mse_loss = mse(net_output, label_img)
        tv_loss = tv(net_output)
        total_loss = mse_loss + 0 * tv_loss  # --------------------

        total_loss.backward()
        optimizer.step()

        if i % opt.print_step == 0:
            print('Evaluate_info: calcuate mse loss {} with {}'.format(net_output.shape, label_img.shape))
            print('Iteration %05d Total loss %f Mse loss %f TV loss %f ' % (
            i, total_loss.data, mse_loss.data, tv_loss.data, ), '\r')

            save_loss_list["mse"].append(mse_loss.data.cpu().numpy())
            save_loss_list["tv"].append(tv_loss.data.cpu().numpy())
            save_loss_list["total"].append(total_loss.data.cpu().numpy())

            if i % opt.draw_step == 0 or i <= opt.draw_step:  # 高频读写文件实在是太影响效率了
                draw_img(opt, label_img, net_input, net_output, i)

            writer.add_image('output', net_output[0].cpu().detach().numpy(), i)
            writer.add_scalars('loss',{'Total loss': total_loss.data,
                                       'Mse loss': mse_loss.data,
                                       'TV loss': tv_loss.data},i)
            writer.add_scalars('learning_rate',{'lr':opt.LR},i)

        if opt.find_best:
            # if training loss improves by at least one percent, we found a new best net
            if best_mse > 1.005 * mse_loss.data:
                best_mse = mse_loss.data
                best_net = copy.deepcopy(model)

    writer.close()
    if opt.find_best:
        model = best_net

    net_out_p0 = model(net_input_const.type(opt.dtype)).data.cpu().numpy()[0][0]

    return save_loss_list, net_out_p0, model


