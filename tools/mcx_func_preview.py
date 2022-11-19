import torch
import numpy as np
from scipy.io import loadmat as load

from reconstruct.mcx_try_function import mcxtry
from reconstruct.train_arguments import TrainArguments


opt = TrainArguments().initialize()

mat_data = load('../test_data/proto_ua/ua/ua_true.mat')
mat_data = mat_data['ua_true']
img_var = np.reshape(mat_data, [256, 256])
ua_true = img_var
img_var = img_var * 1000  # 这一整段的意思就是给出来的图像进行赋值，跑MC
img_var = np.uint8(img_var)
img_var[img_var==0] = 1  # img_var = img_var+1

fai_6 = mcxtry(shuru_image=img_var)  # 得到光通量

fai_nd1 = fai_6[:, :, 127]  # 取其中中间的那一片
fai_nd2 = np.reshape(fai_nd1, [256, 256])
fai_nd3 = fai_nd2 + 1e-12  # 给光通量加一个极小值防止取log时出错

ua_true = ua_true + 1e-12

p0_dl1 = fai_nd3 * ua_true  # 相乘得到p0

p0_dl2 = np.log(p0_dl1)
for iiii in range(256):
    for jjjj in range(256):
        if p0_dl2[iiii][jjjj] < 0:
            p0_dl2[iiii][jjjj] = 0
p0_dl3 = p0_dl2 / np.max(p0_dl2)  # 取log然后归一化与输入的归一化P0相对应

import scipy.io
scipy.io.savemat("./p0_dl315.mat",{'p0_dl315':p0_dl3})

