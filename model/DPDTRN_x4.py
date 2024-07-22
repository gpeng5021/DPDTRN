import torch
import torch.nn as nn
import numpy as np
import math
from model.modules.ecb import ECB
import random
import string
import os
from torchvision import utils as vutils
import torch.nn.functional as F
import functools


def get_upsample_filter(size):
    """Make a 2D bilinear kernel suitable for upsampling"""
    factor = (size + 1) // 2
    if size % 2 == 1:
        center = factor - 1
    else:
        center = factor - 0.5
    og = np.ogrid[:size, :size]
    filter = (1 - abs(og[0] - center) / factor) * \
             (1 - abs(og[1] - center) / factor)
    return torch.from_numpy(filter).float()


def make_layer(block, n_layers):
    layers = []
    for _ in range(n_layers):
        layers.append(block())
    return nn.Sequential(*layers)


class _ResidualDenseBlock_5xConv(nn.Module):
    def __init__(self, nf=64, gc=32, bias=True):
        super(_ResidualDenseBlock_5xConv, self).__init__()
        self.conv1 = nn.Conv2d(nf, gc, 3, 1, 1, bias=bias)
        self.conv2 = nn.Conv2d(nf + gc, gc, 3, 1, 1, bias=bias)
        self.conv3 = nn.Conv2d(nf + 2 * gc, gc, 3, 1, 1, bias=bias)
        self.conv4 = nn.Conv2d(nf + 3 * gc, gc, 3, 1, 1, bias=bias)
        self.conv5 = nn.Conv2d(nf + 4 * gc, nf, 3, 1, 1, bias=bias)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
        x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
        x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), 1)))
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
        return x5 * 0.2 + x


class _RIRD_Block(nn.Module):
    '''Residual in Residual Dense Block'''

    def __init__(self, nf, gc=32):
        super(_RIRD_Block, self).__init__()
        self.RIRD1 = _ResidualDenseBlock_5xConv(nf, gc)
        self.RIRD2 = _ResidualDenseBlock_5xConv(nf, gc)
        self.RIRD3 = _ResidualDenseBlock_5xConv(nf, gc)

    def forward(self, x):
        out = self.RIRD1(x)
        out = self.RIRD2(out)
        out = self.RIRD3(out)
        return out * 0.2 + x


class _CSAM_Module(nn.Module):
    """ Channel-Spatial attention module"""

    def __init__(self, in_dim):
        super(_CSAM_Module, self).__init__()
        self.chanel_in = in_dim

        self.conv = nn.Conv3d(1, 1, 3, 1, 1)
        self.gamma = nn.Parameter(torch.zeros(1))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        m_batchsize, C, height, width = x.size()
        out = x.unsqueeze(1)
        out = self.sigmoid(self.conv(out))
        out = self.gamma * out
        out = out.view(m_batchsize, -1, height, width)
        x = x * out + x
        return x


class _Usm_module(nn.Module):
    def __init__(self, in_channels=64, out_channels=64, nf=64, nb_RIRD=16, gc=32):
        super(_Usm_module, self).__init__()
        RIRD_block_f = functools.partial(_RIRD_Block, nf=nf, gc=gc)
        self.RIRD_trunk = make_layer(RIRD_block_f, nb_RIRD)
        self.trunk_conv = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.usm_conv = nn.Conv2d(nf, nf, 3, 1, 1, bias=False)
        self.csa = _CSAM_Module(out_channels)
        self.ecb_block = ECB(inp_planes=in_channels, out_planes=out_channels, depth_multiplier=2.0, act_type='prelu',
                             with_idt=True)
        self.ecb_factor = 0.01
        self.textures_conv = nn.Conv2d(64, 1, 3, 1, 1, bias=False)

    def forward(self, x, usm_lr):
        trunk = self.trunk_conv(self.RIRD_trunk(x))
        fea = x + trunk

        fea_usm = self.usm_conv(fea - usm_lr)  # todo 相减操作，纹理感知
        fea_uc = self.csa(fea_usm)
        fea_ucla = self.ecb_block(fea_uc) * self.ecb_factor
        fea_end = fea + fea_ucla

        fea_textures = self.textures_conv(fea_ucla)

        return fea_end, fea_textures

    # 做好初始化再放进来  _Usmnet不会放大倍数


class _Usmnet(nn.Module):
    def __init__(self, scale=4, in_channels=64, out_channels=64, nf=64, nb_Usm_module=1, gc=32):
        super(_Usmnet, self).__init__()

        self.Usm_upconv4x = nn.ConvTranspose2d(in_channels=64, out_channels=64, kernel_size=6, stride=4, padding=1,
                                               bias=False)
        self.Usm_module = _Usm_module(in_channels=64, out_channels=64, nf=64, nb_RIRD=16, gc=32)
        self.nb_Usm_module = nb_Usm_module
        self.scale = scale
        self.conv_4usm_lr = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3, stride=1, padding=1,
                                      bias=False)

    def forward(self, x, lr):

        fet = x
        if (self.scale & (self.scale - 1)) == 0:
            fet = self.Usm_upconv4x(fet)

        else:
            raise NotImplementedError

        usm_lr = self.conv_4usm_lr(
            F.interpolate(lr, size=[x.shape[2] * self.scale, x.shape[3] * self.scale], mode='bicubic',
                          align_corners=False))

        for _ in range(self.nb_Usm_module):
            fet, textures = self.Usm_module(fet, usm_lr)
        out_usm = fet

        return out_usm, textures


class Usmsrn(nn.Module):
    def __init__(self, in_c=1, out_c=1, scale=4):
        super(Usmsrn, self).__init__()

        self.scale = scale
        self.conv_first = nn.Conv2d(in_c, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.USMnet = _Usmnet(scale=scale, in_channels=64, out_channels=64, nf=64, nb_Usm_module=1, gc=32)
        self.conv_last = nn.Conv2d(64, out_c, kernel_size=3, stride=1, padding=1, bias=False)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            if isinstance(m, nn.ConvTranspose2d):
                c1, c2, h, w = m.weight.data.size()
                weight = get_upsample_filter(h)
                m.weight.data = weight.view(1, 1, h, w).repeat(c1, c2, 1, 1)
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x):

        in_conv = self.conv_first(x)
        usm_lr = x

        out_usm, textures = self.USMnet(in_conv, usm_lr)
        sr = self.conv_last(out_usm)

        return sr, textures


if __name__ == "__main__":
    from torchsummary import summary

    model = Usmsrn()
    # print(model)
    summary(model, input_size=(1, 64, 64))
