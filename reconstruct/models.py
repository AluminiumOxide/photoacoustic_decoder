import torch.nn as nn
import torch


class Head(nn.Module):
    def __init__(self,channel_in, channel_out):
        super(Head, self).__init__()

        self.head = nn.Sequential(
            nn.ReflectionPad2d(0),
            nn.Conv2d(in_channels=channel_in,out_channels=channel_out,kernel_size=1,stride=1,padding=0,bias=False),
        )

    def forward(self, x):
        x = self.head(x)
        return x


class BoneBlock(nn.Module):
    def __init__(self,channel_in,channel_out,up_sample=True):
        super(BoneBlock, self).__init__()
        self.up_sample = up_sample

        if self.up_sample:
            self.Upsample = nn.Upsample(scale_factor=2, mode='bilinear')

        self.body = nn.Sequential(
            nn.ReLU(),
            nn.BatchNorm2d(num_features=channel_in,affine=True),
            nn.ReflectionPad2d(0),
            nn.Conv2d(in_channels=channel_in, out_channels=channel_out, kernel_size=1, stride=1, padding=0, bias=False),
        )

    def forward(self, x):
        if self.up_sample:
            x = self.Upsample(x)
        x = self.body(x)
        return x


class DeepDecoder(nn.Module):
    def __init__(self, input_channel, num_channels, output_channel):
        super(DeepDecoder,self).__init__()
        # 开始干正事，前面的两层
        self.head = Head(input_channel, num_channels[0])
        self.block_1 = BoneBlock(num_channels[0], num_channels[1])
        self.block_2 = BoneBlock(num_channels[1], num_channels[2])
        self.block_3 = BoneBlock(num_channels[2], num_channels[3])
        self.block_4 = BoneBlock(num_channels[3], num_channels[3])
        self.block_5 = BoneBlock(num_channels[3], output_channel, False)
        self.out = nn.Sigmoid()

    def forward(self, x):
        x = self.head(x)
        x = self.block_1(x)
        x = self.block_2(x)
        x = self.block_3(x)
        x = self.block_4(x)
        x = self.block_5(x)
        x = self.out(x)
        return x


class ResBoneBlock(BoneBlock):  # 直接重写没问题吧!没问题吧!
    def __init__(self,channel_in,channel_out,up_sample=True,end_block=False):
        super(ResBoneBlock, self).__init__(channel_in,channel_out,up_sample=up_sample)

        self.end = end_block

    def forward(self, x):

        if self.up_sample:
            x = self.Upsample(x)
        identity = x
        x = self.body(x)
        if not self.end:
            x += identity

        return x


class DeepDecoderRes(DeepDecoder):
    def __init__(self,input_channel, num_channels, output_channel):
        super(DeepDecoderRes, self).__init__(input_channel,num_channels,output_channel)
        # 直接换block
        self.block_1 = ResBoneBlock(num_channels[0], num_channels[1])
        self.block_2 = ResBoneBlock(num_channels[1], num_channels[2])
        self.block_3 = ResBoneBlock(num_channels[2], num_channels[3])
        self.block_4 = ResBoneBlock(num_channels[3], num_channels[3])
        self.block_5 = ResBoneBlock(num_channels[3], output_channel, False, True)





def decoderrnw(opt):
    model = DeepDecoder(input_channel=opt.input_channel,
                        num_channels=opt.num_channels,
                        output_channel=opt.output_depth
                        )
    return model


def decoderres(opt):
    model = DeepDecoderRes(input_channel=opt.input_channel,
                        num_channels=opt.num_channels,
                        output_channel=opt.output_depth)
    return model



if __name__ == '__main__':
    model = DeepDecoderRes(64,[64,64,64,64],1)
    print(model)