import copy
import os.path
from PIL import Image
import torch
import numpy as np

from torch.autograd import Variable
from torch.utils.tensorboard import SummaryWriter

from reconstruct.Loss import mix_loss, TVLoss
from reconstruct.post_process import draw_img,draw_img_p0

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

        net_input = Variable(torch.zeros(shape))  # net_input随机的储存大小确定并且全部给0
        torch.manual_seed(429)
        net_input.data.uniform_()
        net_input.data *= 1. / 10
        print("InputNoise B0: with shape :{}  and example info :{}".format(net_input.shape, net_input[0][0][0][:8]) )

    net_input_saved = net_input.data.clone()  # 总之是作为输入的噪声
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


def using_mcx(opt, out, mcx_shape):
    """
    :param opt: 字面意思
    :param out: 生成器的输出内容
    :param mcx_shape: 蒙特卡洛模拟空间,如果第一维数据为1,则视为2维数据,  !似乎x,y需要和 out的长宽耦合,之后再说
    举例: [1,256,256]为创建2维平面,对应[占位],x,y
          [3,256,256] 为创建3维空间,对应 x,y,z  # 其中x不能是 1
    :return:
    """
    # if mcx_space[1]==1:  # 哦~为了二维数据
    #     mcx_space = mcx_space[1],mcx_space[2],mcx_space[3]

    amend_list = [out[0, 0, 76, 181],out[0, 0, 127, 127],out[0, 0, 184, 87],out[0, 0, 93, 73],out[0, 0, 146, 194]]
    amend_list = [i.data.cpu().tolist() for i in amend_list]  # 担心一会tensor和数组一起操作出问题
    amend_list = [0.01 if i < 0.01 else i for i in amend_list]  # 嗯

    amend = sum(amend_list)/len(amend_list)
    amend = amend / 0.01  # 这里选出来的像素值对应的ua真值是0.01

    ua_with_gard = out / amend  # 此处out存在梯度值
    ua_true_proto = ua_with_gard.data.cpu()
    ua_true = ua_true_proto.detach().numpy()  # 将ua_true与out分离，使得ua_true不带梯度值方便后续MC运行
    ua_true = np.reshape(ua_true, [256, 256])

    ua_true = ua_true * 1000  # 这一整段的意思就是给出来的图像进行赋值，跑MC
    ua_true = np.uint8(ua_true)

    ua_tune_for_mc = [[100 if i > 100 else 1 if i == 0 else i for i in line] for line in ua_true]
    ua_tune_for_mc = np.array(ua_tune_for_mc)  # 然后我的MC设置的最大的吸收系数是0.1，然后我以0.001作为一个值去跑MC

    fai = mcxtry(input_image=ua_tune_for_mc,mcx_shape=mcx_shape)  # 开始炼丹,并得到光通量

    fai_tune = fai + 1e-8  # 给光通量加一个极小值防止取log时出错
    fai_tune = torch.from_numpy(fai_tune)  # 准备将fai转为tensor数据
    fai_tune = torch.unsqueeze(fai_tune, 0)
    fai_tune = torch.unsqueeze(fai_tune, 0)  # 将fai格式转为1*1*256*256,与网络输出out相对应
    fai_tune = fai_tune.float()  # 将fai数据类型转为float，与网络输出out想对应

    out = out + 1e-8

    p0 = fai_tune.type(opt.dtype) * out  # 相乘得到p0,并算是接上了梯度
    p0_tune = torch.log(p0)

    p0_mask = torch.zeros_like(p0_tune)  # 换了换了，能少些循环就少写循环
    p0_tune = torch.where(p0_tune>0,p0_tune,p0_mask)

    p0_tune = p0_tune / torch.max(p0_tune)  # 取log然后归一化与输入的归一化P0相对应

    p0_tune = p0_tune.type(opt.dtype)  # 其实我感觉这句没用
    print("\tUsing_mcx: return P0 shape {}".format(p0_tune.shape))
    return p0_tune


def nowtime_mcxshape(epoch, total_epoch,x_shape,y_shape):
    """
    :param epoch: 入当前p0的迭代次数
    :param total_epoch: 总次数
    :param x_shape: 输出图像的x和y多大
    :param y_shape:
    :return: 返回目前mcx模拟的空间类似[1,x,y]或[x,y,z]
    """
    final_z = max([x_shape, y_shape])
    rate = 0.6
    if epoch/total_epoch <= rate:
        mcxshape = [1, x_shape, y_shape]
    else:
        nowtime_step = (epoch-total_epoch*rate)/(total_epoch*(1-rate))  # 0~1
        z_shape = int(nowtime_step*nowtime_step*final_z)+2  # int(1/2)向下取整到0,但是我不喜欢用math的函数,就这样吧
        mcxshape = [x_shape, y_shape, z_shape]
    return mcxshape


def evaluate_info(opt, model, input_img, use_mcx=False):
    if use_mcx:
        iter = opt.num_iter_p0
        quantity = 'p0'
    else:
        iter = opt.num_iter_ua
        quantity = 'ua'

    net_input_saved, noise, net_input = load_net_input(opt, input_img)
    optimizer = load_optimizer(opt, model, net_input)
    save_loss_list, mse, tv = load_loss(opt)

    if opt.find_best:
        best_net = copy.deepcopy(model)
        best_mse = 1000000.0

    writer = SummaryWriter(os.path.join(opt.save_path,quantity))
    writer.add_image('org_input',input_img[0].cpu().detach().numpy(),1)

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
            net_input = Variable(net_input_saved + (noise.normal_() * opt.reg_noise_std))

        optimizer.zero_grad()
        out = model(net_input.type(opt.dtype))

        if use_mcx:
            """
            out作为一个1,1,256,256输入不变，mcx_shape根据p0的迭代方法从2维到3维
            使用nowtime_mcxshape()根据当前迭代次数转换mcx的空间
            加判断顺带着变量互换，似乎如果是求p0，后面生成的ua也没有使用的联系？
            """
            mcxshape = nowtime_mcxshape(i, opt.num_iter_p0, x_shape=out.shape[2], y_shape=out.shape[3])
            out_float = using_mcx(opt, out, mcx_shape=mcxshape)
            out, out_float = out_float, out   # out float ua / out p0
        if i % opt.print_step == 0:
            print('Evaluate_info: calcuate mse loss {} with {}'.format(out.shape,input_img.shape))
        mse_loss = mse(out, input_img)
        tv_loss = tv(out)
        total_loss = mse_loss + 0 * tv_loss  # --------------------
        total_loss.backward()
        optimizer.step()

        if i % opt.print_step == 0:
            out2 = model(Variable(net_input_saved).type(opt.dtype))
            loss2 = mse(out2, input_img)

            print('Iteration %05d Total loss %f Mse loss %f TV loss %f Other test %f' % (
            i, total_loss.data, mse_loss.data, tv_loss.data, loss2.data), '\r')

            save_loss_list["mse"].append(mse_loss.data.cpu().numpy())
            save_loss_list["tv"].append(tv_loss.data.cpu().numpy())
            save_loss_list["total"].append(total_loss.data.cpu().numpy())

            if i % opt.draw_step == 0 or i <= opt.draw_step:  # 高频读写文件实在是太影响效率了
                if use_mcx:
                    draw_img_p0(input_img, net_input_saved, net_input, out, out_float)
                else:
                    draw_img(input_img, net_input_saved, net_input, out)
                label_and_output_image = np.array(Image.open('latest_img.png'))[:, :, 0:3]
                writer.add_image('label_and_output', label_and_output_image,i, dataformats='HWC')

            writer.add_image('output', out[0].cpu().detach().numpy(), i)
            writer.add_scalars('loss',{'Total loss': total_loss.data,
                                       'Mse loss': mse_loss.data,
                                       'TV loss': tv_loss.data},i)

        if opt.find_best:
            # if training loss improves by at least one percent, we found a new best net
            if best_mse > 1.005 * mse_loss.data:
                best_mse = mse_loss.data
                best_net = copy.deepcopy(model)

    writer.close()
    if opt.find_best:
        model = best_net

    out_img_np = model(net_input_saved.type(opt.dtype)).data.cpu().numpy()[0][0]

    return save_loss_list, out_img_np, model


if __name__ == '__main__':
    from reconstruct.train_arguments import TrainArguments
    opt = TrainArguments().initialize()
    # p0_epoch = opt.num_iter_p0
    p0_epoch = 1000
    # out作为一个1,1,256,256输入不变，mcx_shape根据p0的迭代方法从2维到3维
    for i in range(p0_epoch):
        out = torch.rand([1,1,256,256],requires_grad=True).type(opt.dtype)
        mcxshape = nowtime_mcxshape(i, p0_epoch, x_shape=out.shape[2],y_shape=out.shape[3])
        print(i,p0_epoch,mcxshape)
        out_float = using_mcx(opt, out,mcx_shape=mcxshape)
        print(out.shape,out_float.shape)