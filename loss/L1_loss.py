#!/usr/bin/env python
# encoding: utf-8

import torch
import torch.nn as nn
import numpy as np
import chainer.functions as F1
import torch.nn.functional as F


class hdrt_loss(nn.Module):
    def __init__(self):
        super(hdrt_loss, self).__init__()

    def forward(self, X, Y, T):
        diff = torch.add(X, -Y)
        y = torch.mul(T, diff)
        y = torch.mean(torch.abs(y))
        return y


class L1_Charbonnier_loss(nn.Module):
    """L1 Charbonnierloss."""

    def __init__(self):
        super(L1_Charbonnier_loss, self).__init__()
        self.eps = 1e-6

    def forward(self, X, Y):
        diff = torch.add(X, -Y)
        error = torch.sqrt(diff * diff + self.eps)
        loss = torch.sum(error)
        return loss


class gradient_loss(nn.Module):
    """gradient_loss."""

    def __init__(self):
        super(gradient_loss, self).__init__()

    def forward(self, X, Y):
        X = X.cpu().detach().numpy()
        Y = Y.cpu().detach().numpy()
        xp = cp.get_array_module(X.data)
        n, c, h, w = X.shape
        wx = xp.array([[[1, -1]]] * c, ndmin=4).astype(xp.float32)
        wy = xp.array([[[1], [-1]]] * c, ndmin=4).astype(xp.float32)

        d_gx = F1.convolution_2d(X, wx)
        d_gy = F1.convolution_2d(X, wy)

        d_tx = F1.convolution_2d(Y, wx)
        d_ty = F1.convolution_2d(Y, wy)

        loss = (F1.sum(F1.absolute(d_gx - d_tx)) + F1.sum(F1.absolute(d_gy - d_ty)))
        # exit()
        return loss
