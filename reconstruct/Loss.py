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


class MixLoss(nn.Module):
    def __init__(self, dtype=torch.cuda.FloatTensor):
        super(MixLoss, self).__init__()
        self.mse = torch.nn.MSELoss().type(dtype)
        self.l1 = nn.L1Loss().type(dtype)
        self.sl1 = nn.SmoothL1Loss(reduction='mean', beta=1.0).type(dtype)
        self.tv = TVLoss().type(dtype)

    def forward(self,output_info,label_info, mask_var=None, apply_f=None):
        # mse loss
        if mask_var is not None:
            mse_loss = self.mse(output_info * mask_var, label_info * mask_var)
        elif apply_f:
            mse_loss = self.mse(apply_f(output_info), label_info)
        else:
            mse_loss = self.mse(output_info, label_info)
        # l1 loss
        l1_loss = self.l1(output_info, label_info)
        # sl1_loss
        sl1_loss = self.sl1(output_info, label_info)
        # tv loss
        tv_loss = self.tv(output_info)

        total_loss = sl1_loss + mse_loss  # + l1_loss + tv_loss
        return total_loss, {'l1': l1_loss.data,
                            'mse': mse_loss.data,
                            'sl1': sl1_loss.data,
                            'tv': tv_loss.data, }


