import copy
import os.path
from PIL import Image
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from torch.autograd import Variable
from torch.utils.tensorboard import SummaryWriter

from reconstruct.Loss import mix_loss, TVLoss
from reconstruct.post_process import draw_img,draw_img_p0

from reconstruct.mcx_try_function import *


def data_gen_ua(show=False):
    array = np.zeros((256, 256))

    # 圆心坐标和半径 -------------------------------------------------
    center_x = 128
    center_y = 128
    radius =100
    x, y = np.meshgrid(np.arange(256), np.arange(256))    # 生成网格
    array[((x - center_x)**2 + (y - center_y)**2) <= radius**2] = 0.01    # 根据圆心和半径的坐标方程设置圆形区域的值

    # 三角形的三个顶点坐标 -----------------------------------------
    vertices = np.array([[100, 100], [200, 100], [150, 200]])
    triangle = Polygon(vertices, closed=True, fill=True)
    for i in range(256):
        for j in range(256):
            if triangle.contains_point((j, i)):
                array[i, j] = 0.02

    # 定义长方形的起点和终点坐标 -----------------------------------------
    x_start, y_start = 65, 60
    x_end, y_end = 95, 190
    array[y_start:y_end, x_start:x_end] = 0.03

    # 定义长方形的起点和终点坐标 -----------------------------------------
    x_start, y_start = 105, 60
    x_end, y_end = 200, 90
    array[y_start:y_end, x_start:x_end] = 0.04

    # 圆心坐标和半径 -------------------------------------------------
    center_x = 150
    center_y = 140
    radius =15
    x, y = np.meshgrid(np.arange(256), np.arange(256))    # 生成网格
    array[((x - center_x)**2 + (y - center_y)**2) <= radius**2] = 0.05    # 根据圆心和半径的坐标方程设置圆形区域的值

    if show:
        plt.imshow(array, cmap='gray')
        plt.colorbar()
        plt.show()

    return array

def data_gen_us(show=False):
    array = np.zeros((256, 256))

    # 圆心坐标和半径 -------------------------------------------------
    center_x = 128
    center_y = 128
    radius =100
    x, y = np.meshgrid(np.arange(256), np.arange(256))    # 生成网格
    array[((x - center_x)**2 + (y - center_y)**2) <= radius**2] = 1    # 根据圆心和半径的坐标方程设置圆形区域的值

    if show:
        plt.imshow(array, cmap='gray')
        plt.colorbar()
        plt.show()

    return array

def simulation_info():
    """
    :return:
    :data_ua_np: ndarray 256,256
    :data_us_np: ndarray 256,256
    :p0_tune:    ndarray 256,256 已经取log并归一化
    """
    data_ua_np = data_gen_ua(False)
    data_us_np = data_gen_us(False)

    ua_3dim = np.expand_dims(data_ua_np, -1).repeat(50, axis=-1)  # 向Z轴扩
    us_3dim = np.expand_dims(data_us_np, -1).repeat(50, axis=-1)  # 向Z轴扩

    ua_4dim = ua_3dim[np.newaxis, :, :, :]
    us_4dim = us_3dim[np.newaxis, :, :, :]

    matrix_100 = np.concatenate((ua_4dim, us_4dim), axis=0)
    matrix_100 = matrix_100.astype(np.float32)

    fai = mcxtry(input_image=matrix_100,shape=[256,256,50],photons=10000000)
    fai = np.where(data_us_np == 0, 0, fai)  # 切割边界
    fai = fai / fai.max()
    p0 = fai * data_ua_np              # 相乘得到p0,并算是接上了梯度
    p0_log = np.log(p0+1e-12)
    p0_log_p = p0_log - p0_log.min()  # 但是这个玩意全是负的 给它拉到正值
    p0_log_1 = p0_log_p / np.max(p0_log_p)  # 取log然后归一化与输入的归一化P0相对应

    # plt.imshow(p0)
    # plt.colorbar()
    # plt.show()

    return data_ua_np,data_us_np,fai,p0_log_1


if __name__ == '__main__':
    sim_ua,sim_us,sim_fai,sim_p0 = simulation_info()

    fig, axes = plt.subplots(1, 4, figsize=(10, 4))
    axes[0].imshow(sim_ua)
    axes[0].set_title('ua setting')
    axes[1].imshow(sim_us)
    axes[1].set_title('us setting')
    axes[2].imshow(sim_fai)
    axes[2].set_title('fi setting')
    axes[3].imshow(sim_p0)
    axes[3].set_title('p0 setting')
    # axes[4].imshow(np.log(sim_fai*sim_ua+1e-12))
    # axes[4].set_title('log(fai*ua+1e-12) setting')
    fig.tight_layout()
    plt.savefig('sim_figure.png')
    # plt.show()


    # sys.exit()

    img_np_p = np.reshape(sim_p0, [1, 256, 256])
    img_p = Variable(torch.from_numpy(img_np_p)[None, :]).type('torch.cuda.FloatTensor')  # 转到GPU可以处理的数据类型
    img_np_a = np.reshape(sim_ua, [1, 256, 256])
    img_a = Variable(torch.from_numpy(img_np_a)[None, :]).type('torch.cuda.FloatTensor')  # 转到GPU可以处理的数据类型




    from reconstruct.Iteration import evaluate_info, pre_evaluate_info
    from reconstruct.post_process import save_mat_info
    from reconstruct.load_info import load_arguments, load_img
    from reconstruct.models import decoderrnw, decoderres


    opt = load_arguments()  # 在这部分加载后续需要的参数
    model = decoderres(opt).type(opt.dtype)  # 实例化模型

    # loss_ua_list, out_img_ua, model = pre_evaluate_info(opt, model, img_a)  # 代替 fit
    # save_mat_info(opt, out_img_ua, loss_ua_list["total"], "ua", True)

    loss_p0_list, out_img_p0, model = evaluate_info(opt, model, img_p)  # 代替 fit
    save_mat_info(opt, out_img_p0, loss_p0_list["total"], "p0", True)










