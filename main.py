from reconstruct.Iteration import evaluate_info
from reconstruct.post_process import save_mat_info
from reconstruct.load_info import load_arguments,load_img
from reconstruct.models import decoderrnw,decoderres

if __name__ == '__main__':
    opt = load_arguments()  # 在这部分加载后续需要的参数
    ua_img_var = load_img(opt.ua_path,opt.dtype)
    p0_img_var = load_img(opt.p0_path,opt.dtype)

    model = decoderres(opt).type(opt.dtype)  # 实例化模型

    # ua pre train
    # loss_ua_list, out_img_ua, model = evaluate_info(opt, model, ua_img_var)  # 代替 fit
    # save_mat_info(opt,out_img_ua,loss_ua_list["total"], "ua", True)

    # p0
    # opt.LR = opt.LR * 0.1
    loss_p0_list, out_img_p0, model = evaluate_info(opt, model, p0_img_var, use_mcx=True)  # 代替 fit
    save_mat_info(opt,out_img_p0,loss_p0_list["total"],"p0", True)