import math

import numpy as np
from struct import unpack
import jdata as jd
from collections import OrderedDict
import sys
import os
import subprocess

import pmcx
import torch


class suppress_stdout_stderr(object):
    '''禁用标准输出和标准错误输入，也就是屏蔽第三方库调用的各种命令行指令 '''

    def __init__(self):
        self.null_fds = [os.open(os.devnull, os.O_RDWR) for x in range(2)]
        self.save_fds = (os.dup(1), os.dup(2))

    def __enter__(self):
        os.dup2(self.null_fds[0], 1)
        os.dup2(self.null_fds[1], 2)

    def __exit__(self, *_):
        os.dup2(self.save_fds[0], 1)
        os.dup2(self.save_fds[1], 2)
        os.close(self.null_fds[0])
        os.close(self.null_fds[1])


def mcxtry(input_image, shape, photons):
    """
    :param input_image: npdarray(2,256,256,z) 对应(ua,x,y,z)和(us,x,y,z)
    :param shape: 蒙特卡洛模拟的空间设置
    :param Photons: 光子包数量
    :return: fai ndarray(256,256)
    """
    x, y, z = shape[0], shape[1], shape[2]
    source_list = [{"Type": "slit", "Pos": [0, 0, math.ceil(z / 2)], "Dir": [1, 0, 0, 0], "Param1": [0, y, 0, 0],
                    "Param2": [0, 0, 0, 0]},
                   {"Type": "slit", "Pos": [0, y, math.ceil(z / 2)], "Dir": [0, -1, 0, 0], "Param1": [x, 0, 0, 0],
                    "Param2": [0, 0, 0, 0]},
                   {"Type": "slit", "Pos": [x, y, math.ceil(z / 2)], "Dir": [-1, 0, 0, 0], "Param1": [0, -y, 0, 0],
                    "Param2": [0, 0, 0, 0]},
                   {"Type": "slit", "Pos": [0, 0, math.ceil(z / 2)], "Dir": [0, 1, 0, 0], "Param1": [x, 0, 0, 0],
                    "Param2": [0, 0, 0, 0]}]
    del x, y, z
    result_list = []
    with suppress_stdout_stderr():
        for source_info in source_list:  # 嗯,先这样
            res = pmcx.run(
                nphoton=photons,  # photons,
                vol=input_image,
                tstart=0,
                tend=5e-9,
                tstep=5e-9,
                gpuid='1',
                autopilot=1,  # 自适应 线程/块配置
                isreflect=1,  # 边界反射
                unitinmm=0.1,
                srctype=source_info["Type"],
                srcpos=source_info["Pos"],
                srcdir=source_info["Dir"],
                srcparam1=source_info["Param1"],
                srcparam2=source_info["Param2"],
                prop=np.array([[0, 0, 1, 1], [0, 1, 0.9, 1.37]]),
            )
            result_list.append(res['flux'])

    results = result_list[0] + result_list[1] + result_list[2] + result_list[3]
    if shape[0] == 1:
        results = np.squeeze(results)  # 二维应该是直接降维吧,中间没调试,我也不知道出来的是256,256还是1,256,256
    else:
        results = results[:, :, math.ceil(shape[2] / 2)-1]  # 如果是三维就取其中中间的那一片
        results = np.squeeze(results)
    return results



def using_mcx(opt, ua, us ,margin, mcx_info, fai_tune_cache):
    """
    :param opt: 字面意思
    :param ua: matrix(1,1,256,256) 吸收系数矩阵`
    :param us: matrix(1,1,256,256) 散射系数矩阵
    :param mcxinfo: 暂定为一个字典，包含 epoch、mcx_shape、mcx_photons
    mcx_photons 光子包数量
    :return:
    :fai_tune: matrix(1,1,256,256) 光通量图像
    :p0_tune:  matrix(1,1,256,256) 初始声压图像
    """
    epoch = mcx_info['epoch']
    mcx_shape = mcx_info['mcx_shape']
    mcx_photons = mcx_info['mcx_photons']

    ua_true = ua.data.cpu().detach()
    ua_true = ua_true.numpy()      # 将ua_true与out分离，使得ua_true不带梯度值方便后续MC运行
    # ua_true[ua_true < opt.margin_flag] = 0
    # ua_true[ua_true >= opt.margin_flag] = opt.ua_min + (opt.ua_max - opt.ua_min) * ( (ua_true[ua_true >= opt.margin_flag] - opt.margin_flag) / (1 - opt.margin_flag) )
    ua_true = ua_true * (opt.ua_max - opt.ua_min) + opt.ua_min

    ua_true = np.reshape(ua_true, [256, 256])


    us_true = us.data.cpu().detach()
    us_true = us_true.numpy()      # 将ua_true与out分离，使得ua_true不带梯度值方便后续MC运行
    us_true = np.reshape(us_true, [256, 256])

    # g_true = np.full_like(us_true, 0.9)  # 暂时的权宜之计，组织全是g=0.9,n=1.37,外部全是g=1,n=1
    # g_true = np.where(us_true == 0, 1, g_true)
    # n_true = np.full_like(us_true, 1.37)
    # n_true = np.where(us_true == 0, 1, n_true)

    ua_3dim = np.expand_dims(ua_true, -1).repeat(mcx_shape[2], axis=-1)  # 向Z轴扩
    us_3dim = np.expand_dims(us_true, -1).repeat(mcx_shape[2], axis=-1)  # 向Z轴扩
    # g_3dim = np.expand_dims(g_true, -1).repeat(mcx_shape[2], axis=-1)  # 向Z轴扩
    # n_3dim = np.expand_dims(n_true, -1).repeat(mcx_shape[2], axis=-1)  # 向Z轴扩

    ua_4dim = ua_3dim[np.newaxis, :, :, :]
    us_4dim = us_3dim[np.newaxis, :, :, :]
    # g_4dim = g_3dim[np.newaxis, :, :, :]
    # n_4dim = n_3dim[np.newaxis, :, :, :]

    matrix_100 = np.concatenate((ua_4dim, us_4dim), axis=0)
    matrix_100 = matrix_100.astype(np.float32)
    # matrix_103 = np.concatenate((ua_4dim, us_4dim, g_4dim, n_4dim), axis=0)
    # matrix_103 = matrix_103.astype(np.uint8)

    if epoch % opt.mcx_step == 0:
        fai = mcxtry(input_image=matrix_100,shape=mcx_shape,photons=mcx_photons)  # 开始炼丹,并得到光通量 (256,256)的ndarray -----------------
        fai = np.where(us_true == 0, 0, fai) # 切割边界
        fai = fai/fai.max()  # 直接将剩余的数值归1,
        fai_tune = torch.from_numpy(fai)  # 准备将fai转为tensor数据
        fai_tune = torch.unsqueeze(fai_tune, 0)
        fai_tune = torch.unsqueeze(fai_tune, 0)  # 将fai格式转为1*1*256*256,与网络输出out相对应
        fai_tune = fai_tune.float()              # 将fai数据类型转为float，与网络输出out想对应
    else:
        fai_tune = fai_tune_cache

    fai_tune = fai_tune.type(opt.dtype)

    p0 = fai_tune * ua                # 相乘得到p0,并算是接上了梯度
    p0_log = torch.log(p0+1e-12)
    p0_log_p = p0_log - p0_log.min()  # 但是这个玩意全是负的 给它拉到正值
    p0_log_1 = p0_log_p / torch.max(p0_log_p)  # 取log然后归一化与输入的归一化P0相对应

    p0_mask = torch.zeros_like(p0_log_1)  # 换了换了，能少些循环就少写循环
    p0_tune = torch.where(p0_log_1>0.0001,p0_log_1,p0_mask)

    if mcx_info['epoch'] % opt.print_step == 0:
        print('\t\tMcx_try_func: P0 output with size{} '.format(fai_tune.shape))
        print("\tUsing_mcx: return P0 shape {}".format(p0_tune.shape))
    return fai_tune,p0_tune




if __name__ == '__main__':
    import numpy as np
    import time
    ua_tune_for_mc = np.random.randint(99, size=(256, 256))+1
    timelist = []
    for i in range(2,256):
        print('step with total z is {} ---'.format(i))
        T1 = time.time()
        fai = mcxtry(input_image=ua_tune_for_mc, shape=[256,256,i],photons=1e6)
        T2 = time.time()
        print('with time {:.4f} s'.format(T2 - T1))
        print('')
        timelist.append(T2 - T1)
        # mcxtry(input_image, shape=[1, 256, 256]):
        # print(fai.shape)

    import matplotlib.pyplot as plt
    plt.plot(timelist)
    plt.show()