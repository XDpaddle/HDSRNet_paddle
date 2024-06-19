import math

import paddle
import paddle.nn as nn
import paddle.nn.functional as F

def default_conv(in_channels, out_channels, kernel_size, bias_attr=True):
    return nn.Conv2D(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size//2), bias_attr=bias_attr)

class MeanShift(nn.Conv2D):
    def __init__(self, rgb_range=255, rgb_mean=(0.4488, 0.4371, 0.4040), rgb_std=(1.0, 1.0, 1.0), sign=-1):
        super(MeanShift, self).__init__(3, 3, kernel_size=1)
        std = paddle.to_tensor(rgb_std)
        self.weight.set_value(paddle.reshape(paddle.eye(3), [3, 3, 1, 1]) / paddle.reshape(std, [3, 1, 1, 1]))
        self.bias = paddle.create_parameter(
            shape=[3],
            dtype='float32',
            default_initializer=paddle.nn.initializer.Assign(sign * rgb_range * paddle.to_tensor(rgb_mean) / std))
        
        self.weight.stop_gradient = True
        self.bias.stop_gradient = True

class BasicBlock(nn.Sequential):
    def __init__(
        self, conv, in_channels, out_channels, kernel_size, stride=1, bias_attr=False,
        bn=True, act=nn.ReLU(True)):

        m = [conv(in_channels, out_channels, kernel_size, bias_attr=bias_attr)]
        if bn:
            m.append(nn.BatchNorm2D(out_channels))
        if act is not None:
            m.append(act)

        super(BasicBlock, self).__init__(*m)

class ResBlock(nn.Layer):
    def __init__(
        self, conv, n_feats, kernel_size,
        bias_attr=True, bn=False, act=nn.ReLU(True), res_scale=1):

        super(ResBlock, self).__init__()
        m = []
        for i in range(2):
            m.append(conv(n_feats, n_feats, kernel_size, bias_attr=bias_attr))
            if bn:
                m.append(nn.BatchNorm2D(n_feats))
            if i == 0:
                m.append(act)

        self.body = nn.Sequential(*m)
        self.res_scale = res_scale

    def forward(self, x):
        res = self.body(x).mul(self.res_scale)
        res += x

        return res

class Upsampler(nn.Sequential):
    def __init__(self, conv, scale, n_feats, bn=False, act=False, bias_attr=True):

        m = []
        if (scale & (scale - 1)) == 0:    # Is scale = 2^n?
            for _ in range(int(math.log(scale, 2))):
                m.append(conv(n_feats, 4 * n_feats, 3, bias_attr))
                m.append(nn.PixelShuffle(2))
                if bn:
                    m.append(nn.BatchNorm2D(n_feats))
                if act == 'relu':
                    m.append(nn.ReLU(True))
                elif act == 'prelu':
                    m.append(nn.PReLU(n_feats))

        elif scale == 3:
            m.append(conv(n_feats, 9 * n_feats, 3, bias_attr))
            m.append(nn.PixelShuffle(3))
            if bn:
                m.append(nn.BatchNorm2D(n_feats))
            if act == 'relu':
                m.append(nn.ReLU(True))
            elif act == 'prelu':
                m.append(nn.PReLU(n_feats))
        else:
            raise NotImplementedError

        super(Upsampler, self).__init__(*m)

