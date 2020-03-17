import torch
import torch.nn as nn
import torch.nn.functional as F


class ContBatchNorm3d(nn.modules.batchnorm._BatchNorm):
    def _check_input_dim(self, input):

        if input.dim() != 5:
            raise ValueError('expected 5D input (got {}D input)'.format(input.dim()))
        #super(ContBatchNorm3d, self)._check_input_dim(input)

    def forward(self, input):
        self._check_input_dim(input)
        return F.batch_norm(
            input, self.running_mean, self.running_var, self.weight, self.bias,
            self.training, self.momentum, self.eps)


class LUConv(nn.Module):
    def __init__(self, in_chan, out_chan, act, groupnorm=True):
        super(LUConv, self).__init__()
        self.conv1 = nn.Conv3d(in_chan, out_chan, kernel_size=3, padding=1, bias=False)
        if groupnorm:
            self.norm = nn.GroupNorm(16, out_chan)
        else:
            self.norm = ContBatchNorm3d(out_chan)

        if act == 'relu':
            self.activation = nn.ReLU(inplace=True)
        elif act == 'prelu':
            self.activation = nn.PReLU()
        elif act == 'elu':
            self.activation = nn.ELU(inplace=True)
        else:
            raise

    def forward(self, x):
        out = self.activation(self.norm(self.conv1(x)))
        return out


def _make_nConv(in_channel, depth, act, double_chnnel=False):
    if double_chnnel:
        layer1 = LUConv(in_channel, 32 * (2 ** (depth+1)),act)
        layer2 = LUConv(32 * (2 ** (depth+1)), 32 * (2 ** (depth+1)),act)
    else:
        layer1 = LUConv(in_channel, 32*(2**depth),act)
        layer2 = LUConv(32*(2**depth), 32*(2**depth)*2,act)

    return nn.Sequential(layer1,layer2)


# class InputTransition(nn.Module):
#     def __init__(self, outChans, elu):
#         super(InputTransition, self).__init__()
#         self.conv1 = nn.Conv3d(1, 16, kernel_size=5, padding=2)
#         self.bn1 = ContBatchNorm3d(16)
#         self.relu1 = ELUCons(elu, 16)
#
#     def forward(self, x):
#         # do we want a PRELU here as well?
#         out = self.bn1(self.conv1(x))
#         # split input in to 16 channels
#         x16 = torch.cat((x, x, x, x, x, x, x, x,
#                          x, x, x, x, x, x, x, x), 1)
#         out = self.relu1(torch.add(out, x16))
#         return out

class DownTransition(nn.Module):
    def __init__(self, in_channel, depth, act):
        super(DownTransition, self).__init__()
        self.ops = _make_nConv(in_channel, depth, act)
        self.maxpool = nn.MaxPool3d(2)
        self.current_depth = depth

    def forward(self, x):
        if self.current_depth == 3:
            out = self.ops(x)
            out_before_pool = out
        else:
            out_before_pool = self.ops(x)
            out = self.maxpool(out_before_pool)
        return out, out_before_pool

class UpTransition(nn.Module):
    def __init__(self, inChans, outChans, depth,act, groupnorm=True, interpolate=False):
        super(UpTransition, self).__init__()
        self.depth = depth
        if interpolate:
            self.up_conv = None
        else:
            self.up_conv = nn.ConvTranspose3d(inChans, outChans, kernel_size=2, stride=2, bias=False)
        if groupnorm:
            self.norm = nn.GroupNorm(8, outChans)
        else:
            self.norm = ContBatchNorm3d(outChans)
        self.ops = _make_nConv(inChans+ outChans//2,depth, act, double_chnnel=True)

    def forward(self, x, skip_x):
        if self.up_conv is None:
            out_up_conv = F.interpolate(x, skip_x.size()[2:], mode='trilinear')
        else:
            out_up_conv = self.norm(self.up_conv(x))
        concat = torch.cat((out_up_conv,skip_x),1)
        out = self.ops(concat)
        return out


class OutputTransition(nn.Module):
    def __init__(self, inChans, n_labels):

        super(OutputTransition, self).__init__()
        self.final_conv = nn.Conv3d(inChans, n_labels, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out = self.sigmoid(self.final_conv(x))
        return out

class UNet3D(nn.Module):
    # the number of convolutions in each layer corresponds
    # to what is in the actual prototxt, not the intent
    def __init__(self, in_channel=1, out_channel=1, act='relu', drop_rate=0.2, interpolate=True):
        super(UNet3D, self).__init__()
        
        self.drop_rate = drop_rate

        self.down_tr64 = DownTransition(in_channel,0,act)
        self.down_tr128 = DownTransition(64,1,act)
        self.down_tr256 = DownTransition(128,2,act)
        self.down_tr512 = DownTransition(256,3,act)

        self.up_tr256 = UpTransition(512, 512,2,act,interpolate=interpolate)
        self.up_tr128 = UpTransition(256,256, 1,act,interpolate=interpolate)
        self.up_tr64 = UpTransition(128,128,0,act,interpolate=interpolate)
        self.out_tr = OutputTransition(64, out_channel)

    def forward(self, x):
        self.out64, self.skip_out64 = self.down_tr64(x)
        self.out128,self.skip_out128 = self.down_tr128(self.out64)
        self.out256,self.skip_out256 = self.down_tr256(self.out128)
        self.out512,self.skip_out512 = self.down_tr512(self.out256)

        if self.training:
            self.out512 = F.dropout3d(self.out512, self.drop_rate)

        self.out_up_256 = self.up_tr256(self.out512, self.skip_out256)
        self.out_up_128 = self.up_tr128(self.out_up_256, self.skip_out128)
        self.out_up_64 = self.up_tr64(self.out_up_128, self.skip_out64)
        self.out = self.out_tr(self.out_up_64)

        return self.out


if __name__ == '__main__':
    import sys
    model = UNet3D()
    print(model)