import os.path
import torch
import numpy as np
import scipy
import time
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt


def save_mat_info(opt=None,image=None,loss=None,name="",show=False):
    # scipy.io.savemat("./save/data_7_13_1.mat", {'data_7_5_1': out_img_ua})
    # scipy.io.savemat("./save/loss_7_13_1.mat", {'loss_7_5_1': loss_ua})
    t = time.localtime()
    save_time = "{}_{}_{}_{}_{}".format(t.tm_year,t.tm_mon,t.tm_mday,t.tm_hour,t.tm_min)

    img_name = name+"_data_"+save_time
    loss_name = name+"_loss_"+save_time
    image_save_path = os.path.join(opt.save_path,img_name)+'.mat'
    loss_save_path = os.path.join(opt.save_path,loss_name)+'.mat'

    scipy.io.savemat(image_save_path, {img_name: image})
    scipy.io.savemat(loss_save_path, {loss_name: loss})

    if show:
        print("\nsave {} image to {}".format(img_name, image_save_path))
        print("save {} loss to {}\n".format(loss_name, loss_save_path))


def flat_tensor(input):
    input_list = input.chunk(input.shape[0],0)
    full_image = False
    line_image = False
    sqrt_input_num = int(pow(len(input_list), 0.5))

    for i, image in enumerate(input_list):
        image = image.squeeze()
        if i % sqrt_input_num == 0: # 每行开头
            if i == sqrt_input_num:  # 第一行不能直接cat，选择复制
                full_image = line_image
            elif i > sqrt_input_num:  # 后面的选择拼接
                full_image = torch.cat((full_image,line_image), 1)
            line_image = image
            continue
        line_image = torch.cat((line_image,image),0)

    full_image = torch.cat((full_image, line_image), 1)

    return full_image


def draw_img(pro_org_img, pro_net_input_saved, pro_net_input, pro_out, if_show_img=False):

    org_img = torch.squeeze(pro_org_img.detach().cpu())
    net_input_saved = torch.squeeze(pro_net_input_saved.cpu().detach())
    net_input = torch.squeeze(pro_net_input.cpu().detach())
    out = torch.squeeze(pro_out.detach().cpu())

    reshape_net_input_saved = flat_tensor(net_input_saved)
    reshape_net_input= flat_tensor(net_input)

    plt.figure(figsize=(32,8))
    plt.subplot(1, 4, 1)

    plt.title('input_image', fontsize=12)
    plt.imshow(org_img)
    plt.tick_params(labelsize=5)

    plt.subplot(1, 4, 2)
    plt.title('net_input_saved', fontsize=12)
    plt.imshow(reshape_net_input_saved)
    plt.tick_params(labelsize=5)

    plt.subplot(1, 4, 3)
    plt.title('net_input(add noise)', fontsize=12)
    plt.imshow(reshape_net_input)
    plt.tick_params(labelsize=5)

    plt.subplot(1, 4, 4)
    plt.title('output_image', fontsize=12)
    plt.imshow(out)
    plt.tick_params(labelsize=5)

    plt.subplots_adjust(bottom=0.15, right=0.88, top=0.85, left=0.08)
    cax = plt.axes([0.9, 0.15, 0.01, 0.7])
    plt.colorbar(cax=cax)

    plt.savefig('latest_img.png')
    if if_show_img:
        plt.show()
    plt.cla()
    plt.close("all")


def draw_img_p0(pro_org_img, pro_net_input_saved, pro_net_input, p0, ua, if_show_img=False):

    org_img = torch.squeeze(pro_org_img.detach().cpu())
    net_input_saved = torch.squeeze(pro_net_input_saved.cpu().detach())
    net_input = torch.squeeze(pro_net_input.cpu().detach())
    out = torch.squeeze(p0.detach().cpu())
    out_ua = torch.squeeze(ua.detach().cpu())

    reshape_net_input_saved = flat_tensor(net_input_saved)
    reshape_net_input= flat_tensor(net_input)

    plt.figure(figsize=(32,8))
    plt.subplot(1, 5, 1)

    plt.title('input_image', fontsize=12)
    plt.imshow(org_img)
    plt.tick_params(labelsize=5)

    plt.subplot(1, 5, 2)
    plt.title('net_input_saved', fontsize=12)
    plt.imshow(reshape_net_input_saved)
    plt.tick_params(labelsize=5)

    plt.subplot(1, 5, 3)
    plt.title('net_input(add noise)', fontsize=12)
    plt.imshow(reshape_net_input)
    plt.tick_params(labelsize=5)

    plt.subplot(1, 5, 4)
    plt.title('output_image_p0', fontsize=12)
    plt.imshow(out)
    plt.tick_params(labelsize=5)

    plt.subplot(1, 5, 5)
    plt.title('output_image_ua', fontsize=12)
    plt.imshow(out_ua)
    plt.tick_params(labelsize=5)

    plt.subplots_adjust(bottom=0.15, right=0.88, top=0.85, left=0.08)
    cax = plt.axes([0.9, 0.15, 0.01, 0.7])
    plt.colorbar(cax=cax)

    plt.savefig('latest_img.png')
    if if_show_img:
        plt.show()
    plt.cla()
    plt.close("all")