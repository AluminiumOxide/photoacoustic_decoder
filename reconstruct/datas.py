import numpy as np
from scipy.io import loadmat as load

import torch
from torch.autograd import Variable


def np_to_var(img_np, dtype = torch.cuda.FloatTensor):
    '''Converts image in numpy.array to torch.Variable.
    From C x W x H [0..1] to  1 x C x W x H [0..1]
    '''
    return Variable(torch.from_numpy(img_np)[None, :])


class LoadMatInfo:
    def __init__(self, option):
        self.opt = option

    def get_ua_img(self):
        mat_data = load(self.opt.ua_path)
        param_name  = self.opt.ua_path.split('/')[-1].split('.mat')[0]
        mat_data = mat_data[param_name]
        mat_data = np.reshape(mat_data, [1, 256, 256])
        img_np = mat_data / np.max(mat_data)  # 将图片进行归一化操作
        img_var = np_to_var(img_np).type(self.opt.dtype)  # 转到GPU可以处理的数据类型

        num_channels = [64] * 4
        output_depth = img_np.shape[0]

        return img_var

    def get_p0_img(self):
        mat_data = load(self.opt.p0_path)
        mat_data = mat_data['p0_p']
        mat_data = np.reshape(mat_data, [1, 256, 256])
        img_np = mat_data / np.max(mat_data)  # 将图片进行归一化操作
        img_var = np_to_var(img_np).type(self.opt.dtype)  # 转到GPU可以处理的数据类型

        num_channels = [64] * 4
        output_depth = img_np.shape[0]

        return img_var
