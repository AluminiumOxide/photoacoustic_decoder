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

# def loadmc2(path, dimension):
#     f = open(path, 'rb')
#     data = f.read()
#     data = unpack('%df' % (len(data) / 4), data)
#     data = np.asarray(data).reshape(dimension, order='F')
#     return data
#
#
# def loadmch(fname, format='f', endian='ieee-le', datadict=False):
#     def fread(fileid, N, Type):
#         if Type == 'c' or Type == 'b' or Type == 'B' or Type == '?':
#             Nb = 1
#         elif Type == 'h' or Type == 'H':
#             Nb = 2
#         elif Type == 'i' or Type == 'I' or Type == 'l' or Type == 'L' or Type == 'f':
#             Nb = 4
#         elif Type == 'q' or Type == 'Q' or Type == 'd':
#             Nb = 8
#         else:
#             raise Exception("Type unknow")
#
#         if N == 1:
#             return unpack(Type, fileid.read(Nb))[0]
#         else:
#             return unpack(str(N) + Type, fileid.read(N * Nb))
#
#     try:
#         fid = open(fname, 'rb')
#     except:
#         raise Exception("Could no open the given file name " + fname)
#
#     data = []
#     header = []
#     photon_seed = []
#
#     while True:
#
#         magicheader = fid.read(4)  # a char is 1 Bytes
#
#         if not magicheader:
#             break
#         elif magicheader != b'MCXH':
#             fid.close()
#             raise Exception("It might not be a mch file!")
#
#         version = fread(fid, 1, 'I')
#
#         assert version == 1, "version higher than 1 is not supported"
#
#         maxmedia = fread(fid, 1, 'I')
#         detnum = fread(fid, 1, 'I')
#         colcount = fread(fid, 1, 'I')
#         totalphoton = fread(fid, 1, 'I')
#         detected = fread(fid, 1, 'I')
#         savedphoton = fread(fid, 1, 'I')
#         unitmm = fread(fid, 1, 'f')
#         seedbyte = fread(fid, 1, 'I')
#         normalizer = fread(fid, 1, 'f')
#         respin = fread(fid, 1, 'i')
#         srcnum = fread(fid, 1, 'I')
#         savedetflag = fread(fid, 1, 'I')
#         junk = fread(fid, 2, 'i')
#
#         detflag = np.asarray(list(bin(savedetflag & (2 ** 8 - 1))[2:]), 'int')
#         if endian == 'ieee-le': detflag = detflag[::-1]  # flip detflag left to right
#         datalen = np.asarray([1, maxmedia, maxmedia, maxmedia, 3, 3, 1])
#         datlen = detflag * datalen[0:len(detflag)]
#
#         dat = fread(fid, (colcount * savedphoton), format)
#         dat = np.asarray(dat).reshape(savedphoton, colcount)
#
#         if savedetflag and len(detflag) > 2 and detflag[2] > 0:
#             dat[:, sum(datlen[0:2]):sum(datlen[0:3])] = dat[:, sum(datlen[0:2]):sum(datlen[0:3])] * unitmm
#         elif savedetflag == 0:
#             dat[:, 1 + maxmedia:(2 * maxmedia)] = dat[:, 1 + maxmedia:(2 * maxmedia)] * unitmm
#
#         # make the data as a dictionary
#         if datadict:
#             if savedetflag:
#                 data_dic = [{} for x in range(savedphoton)]
#                 for photonid in range(savedphoton):
#                     if len(detflag) > 0 and detflag[0] != 0: data_dic[photonid]["detid"] = dat[photonid][0]
#                     if len(detflag) > 1 and detflag[1] != 0: data_dic[photonid]["nscat"] = dat[photonid][
#                                                                                            datlen[0]:1 + datlen[1]]
#                     if len(detflag) > 2 and detflag[2] != 0: data_dic[photonid]["ppath"] = dat[photonid][
#                                                                                            sum(datlen[0:2]):sum(
#                                                                                                datlen[0:3])]
#                     if len(detflag) > 3 and detflag[3] != 0: data_dic[photonid]["mom"] = dat[photonid][
#                                                                                          sum(datlen[0:3]):sum(
#                                                                                              datlen[0:4])]
#                     if len(detflag) > 4 and detflag[4] != 0: data_dic[photonid]["p"] = dat[photonid][
#                                                                                        sum(datlen[0:4]):sum(
#                                                                                            datlen[0:5])]
#                     if len(detflag) > 5 and detflag[5] != 0: data_dic[photonid]["v"] = dat[photonid][
#                                                                                        sum(datlen[0:5]):sum(
#                                                                                            datlen[0:6])]
#                     if len(detflag) > 6 and detflag[6] != 0: data_dic[photonid]["w0"] = dat[photonid][-1]
#
#             elif savedetflag == 0:
#                 data_dic = [{"detid": photon[0],
#                              "nscat": photon[1:1 + maxmedia],
#                              "ppath": photon[1 + maxmedia:1 + 2 * maxmedia],
#                              "mom": photon[1 + 2 * maxmedia:1 + 3 * maxmedia],
#                              "p": photon[-7:-4:1], "v": photon[-4:-1:1],
#                              "w0": photon[-1]} for photon in dat]
#
#             del dat
#             dat = np.asarray(data_dic)
#
#         data.append(dat)
#
#         # if "save photon seed" is True
#         if seedbyte > 0:
#             # seeds = unpack('%dB' % (savedphoton*seedbyte), fid.read(savedphoton*seedbyte))
#             seeds = fread(fid, (savedphoton * seedbyte), 'B')
#             photon_seed.append(np.asarray(seeds).reshape((seedbyte, savedphoton), order='F'))
#
#         if respin > 1: totalphoton *= respin
#
#         header = {'version': version,
#                   'medianum': maxmedia,
#                   'detnum': detnum,
#                   'recordnum': colcount,
#                   'totalphoton': totalphoton,
#                   'detectedphoton': detected,
#                   'savedphoton': savedphoton,
#                   'lengthunit': unitmm,
#                   'seedbyte': seedbyte,
#                   'normalizer': normalizer,
#                   'respin': respin,
#                   'srcnum': srcnum,
#                   'savedetflag': savedetflag}
#
#     fid.close()
#
#     data = np.asarray(data).squeeze()
#
#     if seedbyte > 0:
#         photon_seed = np.asarray(photon_seed).transpose((0, 2, 1)).squeeze()
#
#     return data, header, photon_seed


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
    :return:
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
                nphoton=1000000,  # photons,
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
                prop=np.array([[0, 0, 1, 1], [0.01, 1, 0.9, 1.37]]),
            )
            result_list.append(res['flux'])

    results = result_list[0] + result_list[1] + result_list[2] + result_list[3]
    if shape[0] == 1:
        results = np.squeeze(results)  # 二维应该是直接降维吧,中间没调试,我也不知道出来的是256,256还是1,256,256
    else:
        results = results[:, :, int(shape[2]/2-1)]  # 如果是三维就取其中中间的那一片
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
    # amend_list = [ua[0, 0, 76, 181],ua[0, 0, 127, 127],ua[0, 0, 184, 87],ua[0, 0, 93, 73],ua[0, 0, 146, 194]]
    # amend_list = [i.data.cpu().tolist() for i in amend_list]  # 担心一会tensor和数组一起操作出问题
    # amend_list = [0.01 if i < 0.01 else i for i in amend_list]  # 嗯
    # amend = sum(amend_list)/len(amend_list)
    # amend = amend / 0.01  # 这里选出来的像素值对应的ua真值是0.01
    # ua_with_gard = ua / amend  # 此处out存在梯度值
    # opt.ua_min = 0
    # opt.ua_max = 0.1

    # ua = ua * (opt.ua_max - opt.ua_min) + opt.ua_min
    # ua[margin] = 0
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
        fai = mcxtry(input_image=matrix_100,shape=mcx_shape,photons=mcx_photons)  # 开始炼丹,并得到光通量 -----------------
        # sum_1 = fai.sum()
        fai = np.where(us_true == 0, 0, fai)
        # sum_2 = fai.sum()
        # fai = fai * sum_2 / sum_1
        fai_tune = fai + 1e-8  # 给光通量加一个极小值防止取log时出错
        fai_tune = torch.from_numpy(fai_tune)  # 准备将fai转为tensor数据
        fai_tune = torch.unsqueeze(fai_tune, 0)
        fai_tune = torch.unsqueeze(fai_tune, 0)  # 将fai格式转为1*1*256*256,与网络输出out相对应
        fai_tune = fai_tune.float()              # 将fai数据类型转为float，与网络输出out想对应
    else:
        fai_tune = fai_tune_cache

    fai_tune = fai_tune + 1e-12
    fai_tune = fai_tune.type(opt.dtype)

    p0 = fai_tune * ua              # 相乘得到p0,并算是接上了梯度

    p0_tune = torch.log(p0)

    p0_mask = torch.zeros_like(p0_tune)  # 换了换了，能少些循环就少写循环

    p0_tune = p0_tune / torch.max(p0_tune)  # 取log然后归一化与输入的归一化P0相对应

    p0_tune = torch.where(p0_tune>0.0001,p0_tune,p0_mask)

    p0_tune = p0_tune.type(opt.dtype)  # 其实我感觉这句没用
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