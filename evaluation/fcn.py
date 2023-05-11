import torch.nn as nn
import math
from torch import clamp 
import torch.utils.model_zoo as model_zoo

__all__ = ['fcn', 'UpBlock', 'fcn_res50']

model_urls = {
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth'
}


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

class UpBlock(nn.Module):
    def __init__(self, inplanes, planes, upsample=False):
        super(UpBlock, self).__init__()
        self.conv = nn.Conv2d(inplanes, planes, 1, 1)
        self.bn = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.will_ups = upsample

    def forward(self, x):
        if self.will_ups:
            x = nn.functional.upsample(x, scale_factor=2, mode='bilinear', align_corners=True)

        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)

        return x

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class ResNet(nn.Module):

    def __init__(self, block, layers, nparts):
        self.inplanes = 64
        super(ResNet, self).__init__()

        self.res = nn.Sequential(
                        nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
                        nn.BatchNorm2d(64),
                        nn.ReLU(inplace=True),
                        nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
                        self._make_layer(block, 64, layers[0]),
                        self._make_layer(block, 128, layers[1], stride=2),
                        self._make_layer(block, 256, layers[2], stride=2),
                        self._make_layer(block, 512, layers[3], stride=1),
                        )

        self.decoder = nn.Sequential(
                UpBlock(512*block.expansion, 256, upsample=True),
                UpBlock(256, 128, upsample=True),
                UpBlock(128, 64, upsample=True),
                nn.Conv2d(64, nparts, 1, 1),
                #nn.LeakyReLU()
                )
        
        
    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.res(x)
        z = self.decoder(x)
        return z
    
    def get_encoder_output(self, x):
        x = self.res(x)
        return x

def fcn_res50(pretrained=False, nparts=15):
    model = ResNet(Bottleneck, [3, 4, 6, 3], nparts=nparts)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet50']), strict=False)
    return model