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
    :param input_image: 字面意思
    :param shape: 蒙特卡洛模拟的空间设置
    :param Photons: 嗯,字面意思，光子包数量
    :return:
    """
    # datadict = False
    # if shape[0] == 1:  # 如果这么设置,那就是二维的了
    #     mcx_2d = True  # 后面好像我就没用到这个货
    #     input_image = np.expand_dims(input_image, 0)
    #     tune_image = input_image # .astype('uint8')  # uint8和int32差别大吗？是的很大！相当恐怖！兄弟！
    # else:  # 否则就是三维的了
    # input_image = np.reshape(input_image, [256, 256])  # 我感觉这一步可以删了
    # tune_image = np.expand_dims(input_image, -1).repeat(shape[2], axis=-1)  # .astype('uint8')
    #
    # tune_image_f32 = tune_image.astype(np.float32)
    # tune_image_f32_4dim = tune_image_f32[np.newaxis, :, :, :]

    # tune_us_4dim = np.ones_like(tune_image_f32_4dim)
    # tune_us_f32_4dim = tune_us_4dim.astype(np.float32)
    # tune_image_f32_4dim = np.concatenate((tune_image_f32_4dim, tune_us_f32_4dim), axis=0)
    # cfg = OrderedDict()
    # cfg = {
    #     'Session': {
    #         'ID': 'absorrand',
    #         'Photons': photons   # 1e8  # 真的需要这么多吗？我怎么感觉刚开始训练epoch用1e6也都差不多？还省时间
    #     },
    #     'Forward': {
    #         'T0': 0,
    #         'T1': 5e-09,
    #         'Dt': 5e-09
    #     },
    #     'Domain': {
    #         'MediaFormat': 'byte',
    #         'LengthUnit': 0.1,
    #
    #         'Media': [{"mua": 0,
    #                    "mus": 0,
    #                    "g": 1,
    #                    "n": 1}
    #                   ],
    #
    #         'Dim': shape,  # 这得改啊!不然mc2出问题
    #         'OriginType': 1
    #     },
    #     'Optode': {
    #         'Source': {},  # 光源后面根据情况来
    #         'Detector': []
    #     },
    #     'Shapes': tune_image  # 藏得真深啊!
    # }
    #
    # for i in range(1, 101):  # 调整编号
    #     if i == 1:
    #         cfg_domain_media = {"mua": 0.0, "mus": 0, "g": 1, "n": 1}  # what a delightful bug !
    #     else:
    #         cfg_domain_media = {"mua": i / 1000, "mus": 10, "g": 0.9, "n": 1.37}
    #     cfg["Domain"]["Media"].append(cfg_domain_media)

    # if shape[0] == 1:  # 算了算了,最后改完再统一调
    #     z, x, y = shape[0], shape[1],shape[2]  # z=1
    #     source_list = [{"Type": "slit", "Pos": [0, 0, 0], "Dir": [0, 1, 0], "Param1": [0, 0, y, 0],
    #                     "Param2": [0, 0, 0, 0]},
    #                    {"Type": "slit", "Pos": [0, 0, y], "Dir": [0, 0, -1], "Param1": [0, x, 0, 0],
    #                     "Param2": [0, 0, 0, 0]},
    #                    {"Type": "slit", "Pos": [0, x, y], "Dir": [0, -1, 0], "Param1": [0, 0, -y, 0],
    #                     "Param2": [0, 0, 0, 0]},
    #                    {"Type": "slit", "Pos": [0, 0, 0], "Dir": [0, 0, 1], "Param1": [0, x, 0, 0],
    #                     "Param2": [0, 0, 0, 0]}]
    #
    # else:
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

    # 准备完cfg，开始调用mcx
    # mcxbin = 'mcx'  # 之前的判断应该用不到？mcxlab还能在非win上跑?可能学长得换一下路径
    # mcxbin = "D:\\Alu_proj\\mcx_space\\mcx2023\\bin\\mcx.exe"
    # mcxbin = "D:\\Alu_proj\\mcx_space\\mcx2020\\bin\\mcx.exe"
    # SID = cfg["Session"]["ID"]
    result_list = []
    with suppress_stdout_stderr():
        for source_info in source_list:  # 嗯,先这样
            res = pmcx.run(
                nphoton=10000000,  # photons,
                vol=input_image,
                tstart=0,
                tend=5e-9,
                tstep=5e-9,
                gpuid='1',
                autopilot=1,
                isreflect=1,
                unitinmm=0.1,
                srctype=source_info["Type"],
                srcpos=source_info["Pos"],
                srcdir=source_info["Dir"],
                srcparam1=source_info["Param1"],
                srcparam2=source_info["Param2"],
                prop=np.array([[0, 0, 1, 1], [0.01, 1, 0.9, 1.37]]),
            )
            result_list.append(res['flux'])

            # cfg["Optode"]["Source"] = source_info
            # newdata = cfg.copy()
            # cfg_encoder = jd.encode(newdata, {'compression': 'zlib', 'base64': True})
            # jd.save(cfg_encoder, SID + '.json', indent=4)  # 同名目录下产生 absorrand.json
            #
            # console_content = mcxbin + ' -f ' + SID + '.json' + ' -F mc2 0'
            # # os.system(console_content)  # 同名目录下产生 absorrand.mc2
            # subprocess.check_output(console_content, shell=True)  # 是啊，为什么要用os调用呢
            #
            # if os.path.isfile(SID + '.mch'): mch = loadmch(SID + '.mch', datadict=datadict)
            # if os.path.isfile(SID + '.mc2'):
            #     nbstep = round((cfg["Forward"]["T1"] - cfg["Forward"]["T0"]) / cfg["Forward"]["Dt"])
            #     if "Dim" in cfg["Domain"] and cfg["Domain"]["Dim"] != []:
            #         dt = cfg["Domain"]["Dim"] + [nbstep]
            #     elif "Shapes" in cfg:
            #         for find in cfg["Shapes"]:
            #             if "Grid" in find:
            #                 dt = find["Grid"]["Size"] + [nbstep]
            #
            #     result_list.append(loadmc2(SID + '.mc2', dt))

    results = result_list[0] + result_list[1] + result_list[2] + result_list[3]
    if shape[0] == 1:
        results = np.squeeze(results)  # 二维应该是直接降维吧,中间没调试,我也不知道出来的是256,256还是1,256,256
        # print('\t\tMcx_try_func: P0 output without cut size{} '.format(results.shape))
    else:
        # print('\t\tMcx_try_func: P0 output shape {} cut from half {}'.format(results.shape, int(shape[2] / 2)))
        results = results[:, :, int(shape[2]/ 2)]  # 如果是三维就取其中中间的那一片
        results = np.squeeze(results)
        # print('\t\tMcx_try_func: P0 output cut to {} '.format(results.shape))
    return results



def using_mcx(opt, ua, us , mcx_info, fai_tune_cache):
    """
    :param opt: 字面意思
    :param out: 生成器的输出内容
    :param mcxinfo: 暂定为一个字典，包含 epoch、mcx_shape、mcx_photons
    mcx_shape
    蒙特卡洛模拟空间,如果第一维数据为1,则视为2维数据,  !似乎x,y需要和 out的长宽耦合,之后再说
    举例: [1,256,256]为创建2维平面,对应[占位],x,y
          [3,256,256] 为创建3维空间,对应 x,y,z  # 其中x不能是 1
    mcx_photons
    光子包数量
    :return:
    """
    epoch = mcx_info['epoch']
    mcx_shape = mcx_info['mcx_shape']
    mcx_photons = mcx_info['mcx_photons']

    # amend_list = [out[0, 0, 76, 181],out[0, 0, 127, 127],out[0, 0, 184, 87],out[0, 0, 93, 73],out[0, 0, 146, 194]]
    # amend_list = [i.data.cpu().tolist() for i in amend_list]  # 担心一会tensor和数组一起操作出问题
    # amend_list = [0.01 if i < 0.01 else i for i in amend_list]  # 嗯
    # amend = sum(amend_list)/len(amend_list)
    # amend = amend / 0.01  # 这里选出来的像素值对应的ua真值是0.01
    # ua_with_gard = out / amend  # 此处out存在梯度值

    ua_true = ua.data.cpu().detach()
    ua_true = ua_true.numpy()      # 将ua_true与out分离，使得ua_true不带梯度值方便后续MC运行
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

    ua_true = ua_true + 1e-8
    fai_tune = fai_tune.type(opt.dtype)

    p0 = fai_tune * ua               # 相乘得到p0,并算是接上了梯度

    p0_tune = torch.log(p0)

    p0_mask = torch.zeros_like(p0_tune)  # 换了换了，能少些循环就少写循环
    p0_tune = torch.where(p0_tune>0.0001,p0_tune,p0_mask)

    p0_tune = p0_tune / torch.max(p0_tune)  # 取log然后归一化与输入的归一化P0相对应

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