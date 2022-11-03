import argparse
import torch


class TrainArguments:
    def __init__(self):
        self.parser = argparse.ArgumentParser()

    def initialize(self):
        # self.parser.add_argument('--name', required=True, type=str, default='demo', help='experiment name')
        self.parser.add_argument('--name', type=str, default='demo', help='experiment name')
        # general
        self.parser.add_argument('--device', default='cuda:0', help='device id (i.e. 0 or 0,1 or cpu)')
        self.parser.add_argument('--dtype', default='torch.cuda.FloatTensor', help='')
        # data
        # self.parser.add_argument('--ua_path', default='./test_data/proto_ua/ua/ua_true.mat')
        self.parser.add_argument('--ua_path', default='./test_data/proto_ua/ua/ua_2.mat')
        self.parser.add_argument('--p0_path', default='./test_data/proto_ua/p0/p0_p.mat')
        self.parser.add_argument('--save_path', default='./save/')
        # model
        self.parser.add_argument('--input_channel', type=int, default=64)
        self.parser.add_argument('--num_channels', type=list, default=[64, 64, 64, 64])
        self.parser.add_argument('--output_depth', type=int, default=1)  # img_np.shape[0] 如果是彩色图则为3，灰度图则为1

        self.parser.add_argument('--reg_noise_std', type=float, default=0.001, help='')
        self.parser.add_argument('--reg_noise_decay', type=int, default=500, help='')
        self.parser.add_argument('--num_iter_ua', type=int, default=1000, help='')  # ua的训练次数
        self.parser.add_argument('--num_iter_p0', type=int, default=10, help='')  # p0的训练次数 调用mcx
        self.parser.add_argument('--LR', type=float, default=0.0025, help='')
        self.parser.add_argument('--find_best', type=bool, default=True, help='')

        self.parser.add_argument('--optimizer', type=str, default='adam', help='')
        # others
        self.parser.add_argument('--opt_input', type=bool, default=False, help='')
        self.parser.add_argument('--mask_var', default=None, help='')
        self.parser.add_argument('--apply_f', default=None, help='')
        self.parser.add_argument('--lr_decay_epoch', default=0, help='')
        self.parser.add_argument('--net_input', default=None, help='')
        self.parser.add_argument('--weight_decay', default=0, help='')
        # collect
        self.parser.add_argument('--print_step', default=10, help='')
        self.parser.add_argument('--draw_step', default=100, help='')

        args = self.parser.parse_args()

        device = torch.device(args.device if torch.cuda.is_available() else "cpu")
        return args
