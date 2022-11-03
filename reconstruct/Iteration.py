import copy
import os.path
from PIL import Image
import torch
import numpy as np

from torch.autograd import Variable
from torch.utils.tensorboard import SummaryWriter

from reconstruct.Loss import MixLoss
from reconstruct.post_process import draw_img_ua,draw_img_p0

from reconstruct.mcx_try_function import mcxtry


def load_net_input(opt, input_image):
    if opt.net_input is not None:
        print("input provided")
    else:
        # feed uniform noise into the network
        totalupsample = 2 ** len(opt.num_channels)  # totalupsample=2^4
        width = int(input_image.data.shape[2] / totalupsample)  # 图片的长/上采样次数得到初始随机张量的尺寸
        height = int(input_image.data.shape[3] / totalupsample)
        shape = [1, opt.num_channels[0], width, height]  # shape=[1,64,16,16]
        print("shape: ", shape)
        net_input = Variable(torch.zeros(shape))  # net_input随机的储存大小确定并且全部给0
        net_input.data.uniform_()
        net_input.data *= 1. / 10
        print("B0: ", net_input)

    net_input_saved = net_input.data.clone()
    noise = net_input.data.clone()  # 保存初始张量B0

    return net_input_saved, noise, net_input


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


def exp_lr_scheduler(optimizer, epoch, init_lr=0.001, lr_decay_epoch=500):
    """Decay learning rate by a factor of 0.1 every lr_decay_epoch epochs."""
    lr = init_lr * (0.65 ** (epoch // lr_decay_epoch))

    if epoch % lr_decay_epoch == 0:
        print('LR is set to {}'.format(lr))

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    return optimizer


def using_mcx(opt, out):
    bdzjl1 = out[0, 0, 76, 181]
    if bdzjl1 < 0.01:
        bdzjl1 = 0.01

    bdzjl2 = out[0, 0, 127, 127]
    if bdzjl2 < 0.01:
        bdzjl2 = 0.01

    bdzjl3 = out[0, 0, 184, 87]
    if bdzjl3 < 0.01:
        bdzjl3 = 0.01

    bdzjl4 = out[0, 0, 93, 73]
    if bdzjl4 < 0.01:
        bdzjl4 = 0.01

    bdzjl5 = out[0, 0, 146, 194]
    if bdzjl5 < 0.01:
        bdzjl5 = 0.01

    bdz = (bdzjl1 + bdzjl2 + bdzjl3 + bdzjl4 + bdzjl5) / 5
    bdz1 = bdz / 0.01  # 这里选出来的像素值对应的ua真值是0.01
    out = out / bdz1  # 此处out存在梯度值
    ua_true_proto = out.data.cpu()
    ua_true = ua_true_proto.detach().numpy()  # 将ua_true与out分离，使得ua_true不带梯度值方便后续MC运行
    ua_true = np.reshape(ua_true, [256, 256])

    ua_true = ua_true * 1000  # 这一整段的意思就是给出来的图像进行赋值，跑MC
    ua_true = np.uint8(ua_true)

    for iii in range(256):
        for j in range(256):
            if ua_true[iii][j] > 100:
                ua_true[iii][j] = 100  # 然后我的MC设置的最大的吸收系数是0.1，然后我以0.001作为一个值去跑MC

    # out11 = out.data.cpu().numpy()
    ua_mc = ua_true

    # ua_true,ua_mc1 = uaqutipro(uanor=out,xindex=89,yindex=197)

    # ua_truenp =ua_true.numpy()

    # ua_mc = ua_mc1.detach().numpy()

    fai = mcxtry(shuru_image=ua_mc)  # 得到光通量

    fai_nd1 = fai[:, :, 127]  # 取其中中间的那一片
    fai_nd2 = np.reshape(fai_nd1, [256, 256])
    fai_nd3 = fai_nd2 + 1e-8  # 给光通量加一个极小值防止取log时出错
    fai_nd4 = torch.from_numpy(fai_nd3)  # 准备将fai转为tensor数据
    fai_nd5 = torch.unsqueeze(fai_nd4, 0)
    fai_nd6 = torch.unsqueeze(fai_nd5, 0)  # 将fai格式转为1*1*256*256,与网络输出out相对应
    fai_nd7 = fai_nd6.float()  # 将fai数据类型转为float，与网络输出out想对应
    #             print(fai_nd7.dtype)

    p0_dl1 = fai_nd7.type(opt.dtype) * out  # 相乘得到p0
    p0_dl2 = torch.log(p0_dl1)
    p0_dl3 = p0_dl2 / torch.max(p0_dl2)  # 取log然后归一化与输入的归一化P0相对应

    for iii1 in range(256):
        for j1 in range(256):
            if p0_dl3[0][0][iii1][j1] < 0:
                p0_dl3[0][0][iii1][j1] = 0

    p0_dl3 = p0_dl3.type(opt.dtype)

    return p0_dl3


def evaluate_info(opt, model, input_img, use_mcx=False):
    if use_mcx:
        iter = opt.num_iter_p0
        quantity = 'p0'
    else:
        iter = opt.num_iter_ua
        quantity = 'ua'

    net_input_saved, noise, net_input = load_net_input(opt, input_img)
    optimizer = load_optimizer(opt, model, net_input)
    # save_loss_list, mse, tv = load_loss(opt)
    mix_loss = MixLoss(opt.dtype)
    save_loss_list = {'mse': [], 'tv': [], 'total': [], }

    if opt.find_best:
        best_net = copy.deepcopy(model)
        best_mse = 1000000.0

    writer = SummaryWriter(os.path.join(opt.save_path,quantity))

    writer.add_image('org_input',input_img[0].cpu().detach().numpy(),1)
    for i in range(iter):
        # 学习率衰减
        if opt.lr_decay_epoch != 0:
            optimizer = exp_lr_scheduler(optimizer, i, init_lr=opt.LR, lr_decay_epoch=opt.lr_decay_epoch)
        # 原图添加随即抖动
        if opt.reg_noise_std > 0:
            if i % opt.reg_noise_decay == 0:
                opt.reg_noise_std *= 0.7
            net_input = Variable(net_input_saved + (noise.normal_() * opt.reg_noise_std))

        # def closure():
        optimizer.zero_grad()
        out = model(net_input.type(opt.dtype))

        if use_mcx:
            """
            加判断顺带着变量互换，似乎如果是求p0，后面生成的ua也没有使用的联系？
            """
            out_float = using_mcx(opt, out)
            out, out_float = out_float, out   # out float ua / out p0

        total_loss, loss_info = mix_loss(out, input_img)
        total_loss.backward()
        optimizer.step()

        if i % opt.print_step == 0:
            print('Iteration {:>5d} l1 loss {:.5f} Mse loss {:.5f} Sl1 loss {:.5f} TV loss {:.5f}'
                  .format(i, loss_info['l1'], loss_info['mse'], loss_info['sl1'], loss_info['tv']))

            save_loss_list["mse"].append(loss_info['mse'].cpu().numpy())
            save_loss_list["tv"].append(loss_info['tv'].cpu().numpy())
            save_loss_list["total"].append(total_loss.data.cpu().numpy())

            if i % opt.draw_step == 0 or i <= opt.draw_step:  # 高频读写文件实在是太影响效率了
                if use_mcx:
                    np_figure = draw_img_p0(input_img, net_input_saved, net_input, out, out_float)
                else:
                    np_figure = draw_img_ua(input_img, net_input_saved, net_input, out)
                writer.add_image('label_and_output', np_figure, i, dataformats='HWC')
                writer.add_image('output', out[0].cpu().detach().numpy(), i)
            writer.add_scalars('loss',{'Total loss': total_loss.data, 'L1 loss': loss_info['l1'],
                                       'Mse loss': loss_info['mse'], 'Sl1 loss': loss_info['sl1'],
                                       'TV loss': loss_info['tv'], }, i)

        if opt.find_best:
            # if training loss improves by at least one percent, we found a new best net
            if best_mse > 1.005 * loss_info['mse']:
                best_mse = loss_info['mse']
                best_net = copy.deepcopy(model)

    writer.close()
    if opt.find_best:
        model = best_net

    out_img_np = model(net_input_saved.type(opt.dtype)).data.cpu().numpy()[0][0]

    return save_loss_list, out_img_np, model