import scipy

from reconstruct.Iteration import evaluate_info
from reconstruct.post_process import save_mat_info
from reconstruct.train_arguments import TrainArguments
from reconstruct.datas import LoadMatInfo
from reconstruct.models import decoderrnw,decoderres  # decodernw

if __name__ == '__main__':
    opt = TrainArguments().initialize()  # 在这部分加载后续需要的参数
    loadMatInfo = LoadMatInfo(opt)  # 在这把数据加载完(不对，仅仅准备了一个图)
    # model = decoderrnw(opt).type(opt.dtype)  # 实例化模型
    model = decoderres(opt).type(opt.dtype)  # 实例化模型
    # ua
    ua_img_var = loadMatInfo.get_ua_img()  # 加载图像
    loss_ua_list, out_img_ua, model = evaluate_info(opt, model, ua_img_var)  # 代替 fit
    save_mat_info(opt,out_img_ua,loss_ua_list["total"], "ua", True)

    # p0
    p0_img_var = loadMatInfo.get_p0_img()  # 加载图像
    loss_p0_list, out_img_p0, model = evaluate_info(opt, model, p0_img_var, use_mcx=True)  # 代替 fit
    save_mat_info(opt,out_img_p0,loss_p0_list["total"],"p0", True)


