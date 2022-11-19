import torch
import torch.nn as nn
from torch.autograd import Variable


class TVLoss(nn.Module):
    def __init__(self, TVLoss_weight=1):
        super(TVLoss, self).__init__()
        self.TVLoss_weight = TVLoss_weight

    def forward(self, x):
        batch_size = x.size()[0]
        h_x = x.size()[2]
        w_x = x.size()[3]
        count_h = (x.size()[2] - 1) * x.size()[3]
        count_w = x.size()[2] * (x.size()[3] - 1)
        h_tv = torch.pow((x[:, :, 1:, :] - x[:, :, :h_x - 1, :]), 2).sum()
        w_tv = torch.pow((x[:, :, :, 1:] - x[:, :, :, :w_x - 1]), 2).sum()
        return self.TVLoss_weight * 2 * (h_tv / count_h + w_tv / count_w) / batch_size


def mix_loss(output_info, input_info, dtype=torch.cuda.FloatTensor, mask_var=None, apply_f=None):
    mse = torch.nn.MSELoss().type(dtype)
    tv = TVLoss().type(dtype)
    # mse loss
    if mask_var is not None:
        mse_loss = mse(output_info * mask_var, input_info * mask_var)
    elif apply_f:
        mse_loss = mse(apply_f(output_info), input_info)
    else:
        mse_loss = mse(output_info, input_info)
    # tv loss
    tv_loss = tv(output_info)

    total_loss = mse_loss #  + tv_loss
    return total_loss, mse_loss, tv_loss
