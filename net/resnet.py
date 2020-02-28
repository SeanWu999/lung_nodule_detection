import torch as t
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    def __init__(self, inchannel, outchannel, stride=1):
        super(ResidualBlock, self).__init__()
        self.left = nn.Sequential(
            nn.Conv3d(inchannel, outchannel, kernel_size=3, stride=stride, padding=1),
            nn.BatchNorm3d(outchannel),
            nn.ReLU(inplace=True),
            nn.Conv3d(outchannel, outchannel, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(outchannel)
        )
        self.shortcut = nn.Sequential()
        if stride != 1 or inchannel != outchannel:
            self.shortcut = nn.Sequential(
                nn.Conv3d(inchannel, outchannel, kernel_size=1, stride=stride),
                nn.BatchNorm3d(outchannel)
            )

    def forward(self, x):
        out = self.left(x)
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class ResNet(nn.Module):
    def __init__(self, ResidualBlock, num_classes=2):
        super(ResNet, self).__init__()
        self.inchannel = 8
        self.conv1 = nn.Sequential(
            nn.Conv3d(1, 8, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(8),
            nn.ReLU(),
        )
        self.layer1 = self.make_layer(ResidualBlock, 8, 2, stride=1)
        self.layer2 = self.make_layer(ResidualBlock, 16, 2, stride=2)
        self.layer3 = self.make_layer(ResidualBlock, 32, 2, stride=2)
        self.layer4 = self.make_layer(ResidualBlock, 64, 2, stride=2)
        self.layer5 = self.make_layer(ResidualBlock, 128, 2, stride=2)
        self.fc = nn.Linear(128, num_classes)
        #self.activate = nn.Sigmoid()

    def make_layer(self, block, outchannels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)   #strides=[1,1] !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        layers = []
        for stride in strides:
            layers.append(block(self.inchannel, outchannels, stride))
            self.inchannel = outchannels
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer5(out)
        out = F.avg_pool3d(out, 3)
        #view(start=,stop=),将多行拼接成一行，flat
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        #out = self.activate(out)
        #out = F.softmax(out,dim=1)
        return out



def ResNet18():
    return ResNet(ResidualBlock)



if __name__ == "__main__":
    input = t.autograd.Variable(t.randn(1, 1, 48, 48, 48))
    num_params = 0
    model=ResNet18()
    for p in model.parameters():
        num_params += p.numel()
    print("The number of parameters: [{}]".format(num_params))
    output=model(input)
    print(output)
