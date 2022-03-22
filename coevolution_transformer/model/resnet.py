# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
import torch
from torch import nn
import torch.nn.functional as F


def bn_conv2d(in_planes, out_planes, kernel_size, dilated, bias):
    return nn.Sequential(
        nn.Conv2d(
            in_planes,
            out_planes,
            kernel_size=kernel_size,
            dilation=dilated,
            padding=(dilated * (kernel_size - 1) + 1) // 2,
            bias=bias,
        ),
        nn.BatchNorm2d(out_planes)
    )


class BasicBlock2D(nn.Module):
    def __init__(self, inplanes, planes, dilated, bias=False, dropout=0):
        super(BasicBlock2D, self).__init__()
        self.inplanes = inplanes
        self.planes = planes
        self.conv1 = bn_conv2d(inplanes, planes, 3, dilated, bias)
        self.dropout = nn.Dropout(dropout, inplace=True)
        self.conv2 = bn_conv2d(planes, planes, 3, dilated, bias)
        if inplanes != planes:
            self.projection = bn_conv2d(inplanes, planes, 1, 1, bias)

    def forward(self, x):
        residual = x
        out = F.relu(self.conv1(x), inplace=True)
        out = self.conv2(out)
        out = self.dropout(out)
        if self.inplanes != self.planes:
            residual = self.projection(residual)
        out += residual
        out = F.relu(out, inplace=True)

        return out


def make_layers(planes, dilated, repeats, block=BasicBlock2D):
    layers = []
    for i in range(repeats):
        layers.append(block(planes, planes, dilated[i % len(dilated)]))
    return nn.Sequential(*layers)
