# coding:utf8
import torch as t
import torch.nn as nn
import torch.nn.functional as F
from layers import Inception_v1, Inception_v2, BasicConv, Deconv, SingleConv, downsample

class Segmentation(nn.Module):
    def __init__(self):
        super(Segmentation, self).__init__()
        self.model_name = "seg"
        self.basic_depth = 16
        self.conv1 = BasicConv(1, self.basic_depth)  # (64，64，64)
        self.downsample1 = Inception_v1(self.basic_depth, self.basic_depth)  # (32,32,32)
        self.conv2 = BasicConv(self.basic_depth, 2*self.basic_depth)  # (32,32,32)
        self.downsample2 = Inception_v1(2*self.basic_depth, 2*self.basic_depth)  # (16,16,16)

        self.conv3 = SingleConv(2*self.basic_depth,1*self.basic_depth)
        self.incept3 = Inception_v2(1*self.basic_depth, 1*self.basic_depth)
        self.deconv3 = Deconv(1*self.basic_depth, 1*self.basic_depth)

        self.conv4 = SingleConv(3*self.basic_depth, 1*self.basic_depth)
        self.incept4 = Inception_v2(1*self.basic_depth, 1*self.basic_depth)
        self.deconv4 = Deconv(1*self.basic_depth, 1*self.basic_depth)

        self.conv5 = nn.Conv3d(2*self.basic_depth, 1, 1)
        self.activate = nn.Sigmoid()

    def forward(self, x):
        conv1 = self.conv1(x)  # (1,16,48,48,48)
        down1 = self.downsample1(conv1)  # (1,16,24,24,24)

        conv2 = self.conv2(down1)   # (1,32,24,24,24)
        down2 = self.downsample2(conv2)  # (1,32,12,12,12)

        conv3 = self.incept3(self.conv3(down2)) #(1,16,12,12,12)
        up3 = self.deconv3(conv3)  # (1,16,24,24,24)
        up3 = t.cat((up3, conv2), 1) #(1,48,24,24)

        conv4 = self.incept4(self.conv4(up3)) #(1,16,24,24,24)
        up4 = self.deconv4(conv4)  # (1,16,48,48,48)
        up4 = t.cat((up4, conv1), 1) #(1,32,48,48,48)

        conv5 = self.conv5(up4)
        return self.activate(conv5)


if __name__ == "__main__":
    a = t.tensor(t.randn(1, 1, 48, 48, 48))
    model = Segmentation()
    b = model(a)
    print(b.shape)