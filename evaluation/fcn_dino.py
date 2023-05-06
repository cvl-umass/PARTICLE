import torch.nn as nn
import math
from torch import clamp 
import torch.utils.model_zoo as model_zoo
import torch
from vision_transformer import vit_small

__all__ = ['fcn', 'UpBlock', 'fcn_dino']


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

class UpBlock(nn.Module):
    def __init__(self, inplanes, planes, upsample=False):
        super(UpBlock, self).__init__()
        self.conv = nn.Conv2d(inplanes, planes, 3, 1, 1)
        self.bn = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, 3, 1, 1)
        self.bn2 = nn.BatchNorm2d(planes)
        self.will_ups = upsample

    def forward(self, x):
        if self.will_ups:

            x = nn.functional.upsample(x, scale_factor=2, mode='bilinear', align_corners=True)

        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)

        return x

class DinoSeg(nn.Module):

    def __init__(self, nparts):
        self.inplanes = 64
        super(DinoSeg, self).__init__()
        # self.backbone = torch.hub.load('facebookresearch/dino:main', 'dino_vits8')
        self.backbone = vit_small(patch_size=8)

        self.decoder = nn.Sequential(
                UpBlock(64, 256, upsample=True),
                # UpBlock(256, 256, upsample=True),
                UpBlock(256, 128, upsample=True),
                UpBlock(128, 64, upsample=True),
                nn.Conv2d(64, nparts, 1, 1),
                )


    def forward(self, x):
        x, att = self.backbone.get_intermediate_layers(x, 4)
        x = x[-1][:,0][:,1:]
        x = x.permute(0,2,1)
        x = x.reshape(-1,64,28,28)
        z = self.decoder(x)

        return z


def fcn_dino(pretrained=False, nparts=19):
    model = DinoSeg(nparts=nparts)
    return model
