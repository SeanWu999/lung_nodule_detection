# coding:utf8
from __future__ import print_function
import argparse
import torch
import torch.nn as nn
from collections import OrderedDict
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.autograd import Variable

class Deconv(nn.Module):
    def __init__(self, cin, cout, keep_depth=False):
        super(Deconv, self).__init__()
        if keep_depth:
            kernel_stride=[2,2,1]
        else:
            kernel_stride=2
        self.model = nn.Sequential(OrderedDict([
            ('deconv1', nn.ConvTranspose3d(cin, cout, kernel_size=kernel_stride, stride=kernel_stride)),
            ('norm', nn.BatchNorm3d(cout)),
            ('relu', nn.ReLU(inplace=True)),
        ]))

    def forward(self, x):
        return self.model(x)

class downsample(nn.Module):
    def __init__(self, cin, cout, padding=1):
        super(downsample, self).__init__()
        self.padding = padding
        self.model = nn.Sequential(OrderedDict([
            ('conv1_1', nn.Conv3d(cin, cout, 3, padding=self.padding)),
            ('pool_1', nn.MaxPool3d(kernel_size=2, stride=2, return_indices=False, ceil_mode=False)),
            ('norm1_1', nn.BatchNorm3d(cout)),
            ('relu1_1', nn.ReLU(inplace=True)),
        ]))

    def forward(self, x):
        return self.model(x)

class SingleConv(nn.Module):
    def __init__(self, cin, cout, padding=1):
        super(SingleConv, self).__init__()
        self.padding = padding
        self.model = nn.Sequential(OrderedDict([
            ('conv1_1', nn.Conv3d(cin, cout, 3, padding=self.padding)),
            ('norm1_1', nn.BatchNorm3d(cout)),
            ('relu1_1', nn.ReLU(inplace=True)),
        ]))

    def forward(self, x):
        return self.model(x)

class BasicConv(nn.Module):
    def __init__(self, cin, cout):
        super(BasicConv, self).__init__()
        self.model = nn.Sequential(OrderedDict([
            ('conv1_1', nn.Conv3d(cin, cout, 3, padding=1)),
            ('norm1_1', nn.BatchNorm3d(cout)),
            ('relu1_1', nn.ReLU(inplace=True)),
            ('conv1_2', nn.Conv3d(cout, cout, 3, padding=1)),
            ('norm1_2', nn.BatchNorm3d(cout)),
            ('relu1_2', nn.ReLU(inplace=True)),
        ]))

    def forward(self, x):
        return self.model(x)

class Inception_v1(nn.Module):
    def __init__(self, cin, co, relu=True, norm=True, keep_depth = False):
        super(Inception_v1, self).__init__()
        assert (co % 4 == 0)
        cos = int(co / 4)
        self.activa = nn.Sequential()
        if norm:
            self.activa.add_module('norm', nn.BatchNorm3d(co))
        if relu:
            self.activa.add_module('relu', nn.ReLU(True))
        if keep_depth:
            steps=[2,2,1]
            pools=[2,2,1]
        else:
            pools=2
            steps=2
        self.branch1 = nn.Conv3d(cin, cos, 1, stride=steps)

        self.branch2 = nn.Sequential(OrderedDict([
            ('conv1', nn.Conv3d(cin, 2 * cos, 1)),
            ('norm1', nn.BatchNorm3d(2 * cos)),
            ('relu1', nn.ReLU(inplace=True)),
            ('conv2', nn.Conv3d(2 * cos, cos, 3, stride=steps, padding=1)),
        ]))
        self.branch3 = nn.Sequential(OrderedDict([
            ('conv1', nn.Conv3d(cin, 2 * cos, 1, stride=1)),
            ('norm1', nn.BatchNorm3d(2 * cos)),
            ('relu1', nn.ReLU(inplace=True)),
            ('conv2', nn.Conv3d(2 * cos, cos, 5, stride=steps, padding=2)),
        ]))

        self.branch4 = nn.Sequential(OrderedDict([
            ('pool', nn.MaxPool3d(pools)),
            ('conv', nn.Conv3d(cin, cos, 1, stride=1))
        ]))

    def forward(self, x):
        branch1 = self.branch1(x)
        branch2 = self.branch2(x)
        branch3 = self.branch3(x)
        branch4 = self.branch4(x)
        result = torch.cat((branch1, branch2, branch3, branch4), 1)
        return self.activa(result)

class Inception_v2(nn.Module):
    def __init__(self, cin, co, relu=True, norm=True):
        super(Inception_v2, self).__init__()
        assert (co % 4 == 0)
        cos = int(co / 4)
        self.activa = nn.Sequential()
        if norm: self.activa.add_module('norm', nn.BatchNorm3d(co))
        if relu: self.activa.add_module('relu', nn.ReLU(True))

        self.branch1 = nn.Conv3d(cin, cos, 1)

        self.branch2 = nn.Sequential(OrderedDict([
            ('conv1', nn.Conv3d(cin, 2 * cos, 1)),
            ('norm1', nn.BatchNorm3d(2 * cos)),
            ('relu1', nn.ReLU(inplace=True)),
            ('conv2', nn.Conv3d(2 * cos, cos, 3, stride=1, padding=1)),
        ]))
        self.branch3 = nn.Sequential(OrderedDict([
            ('conv1', nn.Conv3d(cin, 2 * cos, 1, stride=1)),
            ('norm1', nn.BatchNorm3d(2 * cos)),
            ('relu1', nn.ReLU(inplace=True)),
            ('conv2', nn.Conv3d(2 * cos, cos, 5, stride=1, padding=2)),
        ]))

        self.branch4 = nn.Sequential(OrderedDict([
            ('pool', nn.MaxPool3d(3, stride=1, padding=1)),
            ('conv', nn.Conv3d(cin, cos, 1, stride=1))
        ]))

    def forward(self, x):
        branch1 = self.branch1(x)
        branch2 = self.branch2(x)
        branch3 = self.branch3(x)
        branch4 = self.branch4(x)
        result = torch.cat((branch1, branch2, branch3, branch4), 1)
        return self.activa(result)







