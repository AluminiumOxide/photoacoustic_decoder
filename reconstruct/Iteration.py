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

        net_const = torch.tensor(torch.zeros(shape)).requires_grad_()  # net_const 随机的储存大小确定并且全部给0
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


def exp_lr_scheduler(optimizer, epoch, init_lr=0.001, lr_decay_epoch=500):
    """Decay learning rate by a factor of 0.1 every lr_decay_epoch epochs."""
    lr = init_lr * (0.65 ** (epoch // lr_decay_epoch))

    if epoch % lr_decay_epoch == 0:
        print('LR is set to {}'.format(lr))

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    return optimizer


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

    mcxshape = [1, x_shape, y_shape]
    # mcxshape = [x_shape, y_shape,50]
    return mcxshape


def nowtime_photons(epoch, total_epoch):
    begin_photon = 1e5
    end_photon = 5e7
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


def evaluate_info(opt, model, label_img, use_mcx=False):
    if use_mcx:
        iter = opt.num_iter_p0
        quantity = 'p0'
    else:
        iter = opt.num_iter_ua
        quantity = 'ua'

    net_input_const = load_net_input(opt, label_img)
    optimizer = load_optimizer(opt, model, net_input_const)
    save_loss_list, mse, tv = load_loss(opt)

    if opt.find_best:
        best_net = copy.deepcopy(model)
        best_mse = 1000000.0

    writer = SummaryWriter(os.path.join(opt.save_path,quantity))
    writer.add_image('org_input', label_img[0].cpu().detach().numpy(), 1)

    mc_out = False
    for i in range(iter):  # --------------------------------------------------------------------------------
        if i % opt.print_step == 0:
            print('\nEvaluate_info: {} epoch {}'.format(quantity, i))
        # 学习率衰减
        if opt.lr_decay_epoch != 0:
            optimizer = exp_lr_scheduler(optimizer, i, init_lr=opt.LR, lr_decay_epoch=opt.lr_decay_epoch)
        # 原图添加随即抖动
        if opt.reg_noise_std > 0:
            if i % opt.reg_noise_decay == 0:
                opt.reg_noise_std *= 0.7
            net_input = net_input_const + torch.randn(net_input_const.shape) * opt.reg_noise_std

        optimizer.zero_grad()
        net_output = model(net_input.type(opt.dtype))

        if use_mcx:
            """
            out作为一个1,1,256,256输入不变，mcx_shape根据p0的迭代方法从2维到3维
            使用nowtime_mcxshape()根据当前迭代次数转换mcx的空间
            加判断顺带着变量互换，似乎如果是求p0，后面生成的ua也没有使用的联系？
            """
            mcxinfo = nowtime_mcxinfo(i, opt.num_iter_p0, x_shape=net_output.shape[2], y_shape=net_output.shape[3])
            if i % opt.print_step == 0:
                print('Evaluate_info: with mcx info {}'.format(mcxinfo))
            # mc_out, mc_mix = using_mcx(opt, net_output, mcx_info=mcxinfo)  # 输出的分别是 fai 和 p0
            mc_out, mc_mix = using_mcx(opt, net_output, mcx_info=mcxinfo,fai_tune_cache=mc_out)  # 输出的分别是 fai 和 p0
            """ 调整后
            :param net_output: 网络输出 光子吸收系数分布 ua
            :param mc_out: 蒙卡输出 光通量分布 fai
            :param mc_mix: 蒙卡+网络计算得到的 初始声压图像 p0
            """
            # Huber_loss = nn.SmoothL1Loss(beta=1)
            mse_loss = mse(mc_mix, label_img)
            tv_loss = tv(mc_mix)
            total_loss = mse_loss + 0 * tv_loss  # --------------------
        else:
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
                if use_mcx:
                    draw_img_p0(opt, label_img, net_input, mc_out ,mc_mix, net_output, i)
                else:
                    draw_img(opt, label_img, net_input, net_output, i)
                # label_and_output_image = np.array(Image.open('latest_img.png'))[:, :, 0:3]
                # writer.add_image('label_and_output', label_and_output_image,i, dataformats='HWC')

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


