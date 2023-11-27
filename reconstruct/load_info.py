import argparse
import os

import numpy as np
from scipy.io import loadmat as load

import torch
from torch.autograd import Variable


def load_arguments():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    parser = argparse.ArgumentParser()

    parser.add_argument('--name', type=str, default='demo', help='experiment name')
    # general
    parser.add_argument('--device', default=device, help='device id (i.e. 0 or 0,1 or cpu)')
    parser.add_argument('--dtype', default='torch.cuda.FloatTensor', help='')
    # data
    # parser.add_argument('--ua_path', default='./test_data/proto_ua_agar/ua/ua_2.mat')
    # parser.add_argument('--p0_path', default='./test_data/proto_ua_agar/p0/p0_p.mat')  # p0_final
    parser.add_argument('--ua_path', default='./test_data/proto_ua_liver/ua/ua.mat')
    parser.add_argument('--p0_path', default='./test_data/proto_ua_liver/p0/p0.mat')  # p0_final
    parser.add_argument('--save_path', default='./save/lyp11_16/')
    # model
    parser.add_argument('--input_channel', type=int, default=64)
    parser.add_argument('--num_channels', type=list, default=[64, 64, 64, 64])
    parser.add_argument('--output_depth', type=int, default=1)  # img_np.shape[0] 如果是彩色图则为3，灰度图则为1
    parser.add_argument('--reg_noise_std', type=float, default=0.001, help='')
    parser.add_argument('--reg_noise_decay', type=int, default=500, help='')
    parser.add_argument('--num_iter_ua', type=int, default=101, help='')  # 相当于预训练
    parser.add_argument('--num_iter_p0', type=int, default=int(10000+1), help='')  # 调用mcx
    parser.add_argument('--LR', type=float, default=0.025, help='')
    parser.add_argument('--find_best', type=bool, default=True, help='')
    parser.add_argument('--optimizer', type=str, default='adam', help='')
    # others
    parser.add_argument('--opt_input', type=bool, default=False, help='')
    parser.add_argument('--mask_var', default=None, help='')
    parser.add_argument('--apply_f', default=None, help='')
    parser.add_argument('--lr_decay_epoch', default=3000, help='')
    parser.add_argument('--net_input', default=None, help='')
    parser.add_argument('--weight_decay', default=0, help='')
    # collect
    parser.add_argument('--mcx_step', default=50, help='进行多少次迭代执行一次mcx')
    parser.add_argument('--print_step', default=50, help='进行多少次迭代打印一次命令行')
    parser.add_argument('--draw_step', default=50, help='进行多少次迭代进行一次绘图')


    args = parser.parse_args()

    if not os.path.exists(args.save_path):
        # 如果目录不存在，使用os.makedirs()函数创建目录
        os.makedirs(args.save_path)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    return args


def np_to_var(img_np, dtype = torch.cuda.FloatTensor):
    '''Converts image in numpy.array to torch.Variable.
    From C x W x H [0..1] to  1 x C x W x H [0..1]
    '''
    return Variable(torch.from_numpy(img_np)[None, :])


def load_img(img_path,dtype='torch.cuda.FloatTensor'):
    mat_data = load(img_path)
    param_name  = img_path.split('/')[-1].split('.mat')[0]
    mat_data = mat_data[param_name]
    img_np = np.reshape(mat_data, [1, 256, 256])
    # img_np = mat_data / np.max(mat_data)  # 归一化
    img_var = np_to_var(img_np).type(dtype)  # 转到GPU可以处理的数据类型
    return img_var
