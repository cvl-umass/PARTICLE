import torch
import torch.nn as nn
import math
import numpy as np
import torch.utils.model_zoo as model_zoo

__all__ = ['ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101',
           'resnet152']

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class Normalize(nn.Module):

    def __init__(self, power=2):
        super(Normalize, self).__init__()
        self.power = power

    def forward(self, x):
        norm = x.pow(self.power).sum(1, keepdim=True).pow(1. / self.power)
        out = x.div(norm)
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


class ResNet(nn.Module):

    def __init__(self, block, layers, low_dim=128, in_channel=3, width=1, pool_size=7):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channel, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)

        self.base = int(64 * width)

        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, self.base, layers[0])
        self.layer2 = self._make_layer(block, self.base * 2, layers[1], stride=2)
        self.layer3 = self._make_layer(block, self.base * 4, layers[2], stride=2)
        self.layer4 = self._make_layer(block, self.base * 8, layers[3], stride=2)


        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

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

    def forward(self, x, layer=7, use_hypercol=False, output_shape=(48,48)):
        if layer <= 0:
            return x
        x1 = self.conv1(x)
        x2 = self.bn1(x1)
        x3 = self.relu(x2)
        x4 = self.maxpool(x3)
        x5 = self.layer1(x4)
        x6 = self.layer2(x5)
        x7 = self.layer3(x6)
        x8 = self.layer4(x7)

        return x4, x5, x6, x7, x8


def resnet18(pretrained=False, **kwargs):
    """Constructs a ResNet-18 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet18']))
    return model


def resnet34(pretrained=False, **kwargs):
    """Constructs a ResNet-34 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet34']))
    return model


def resnet50(pretrained=False, **kwargs):
    """Constructs a ResNet-50 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    model = nn.Sequential(*list(model.children()))
    state_dict = torch.load('./pretrained_models/resnet50_detcon.pth')['state_dict']
    model.load_state_dict(state_dict,strict=True)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet50']))
    return model


def resnet101(pretrained=False, **kwargs):
    """Constructs a ResNet-101 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 23, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet101']))
    return model


def resnet152(pretrained=False, **kwargs):
    """Constructs a ResNet-152 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 8, 36, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet152']))
    return model


class InsResNet50(nn.Module):
    """Encoder for instance discrimination and MoCo"""
    def __init__(self, width=1, pool_size=7):
        super(InsResNet50, self).__init__()
        self.encoder = resnet50(width=width, pool_size=pool_size)

    def forward(self, x, layer=7, use_hypercol=False, output_shape=(48,48)):
        return self.encoder(x, layer, use_hypercol, output_shape)


class InsResNet18(nn.Module):
    """Encoder for instance discrimination and MoCo"""
    def __init__(self, width=1, pool_size=7):
        super(InsResNet18, self).__init__()
        self.encoder = resnet18(width=width, pool_size=pool_size)
        self.encoder = nn.DataParallel(self.encoder)

    def forward(self, x, layer=7, use_hypercol=False, output_shape=(48,48)):
        return self.encoder(x, layer, use_hypercol, output_shape)


class InsResNet34(nn.Module):
    """Encoder for instance discrimination and MoCo"""
    def __init__(self, width=1, pool_size=7):
        super(InsResNet34, self).__init__()
        self.encoder = resnet34(width=width, pool_size=pool_size)
        self.encoder = nn.DataParallel(self.encoder)

    def forward(self, x, layer=7, use_hypercol=False, output_shape=(48,48)):
        return self.encoder(x, layer, use_hypercol, output_shape)


class InsResNet101(nn.Module):
    """Encoder for instance discrimination and MoCo"""
    def __init__(self, width=1, pool_size=7):
        super(InsResNet101, self).__init__()
        self.encoder = resnet101(width=width, pool_size=pool_size)
        self.encoder = nn.DataParallel(self.encoder)

    def forward(self, x, layer=7, use_hypercol=False, output_shape=(48,48)):
        return self.encoder(x, layer, use_hypercol, output_shape)


class InsResNet152(nn.Module):
    """Encoder for instance discrimination and MoCo"""
    def __init__(self, width=1, pool_size=7):
        super(InsResNet152, self).__init__()
        self.encoder = resnet152(width=width, pool_size=pool_size)
        self.encoder = nn.DataParallel(self.encoder)

    def forward(self, x, layer=7, use_hypercol=False, output_shape=(48,48)):
        return self.encoder(x, layer, use_hypercol, output_shape)


class ResNetV1(nn.Module):
    def __init__(self, name='resnet50'):
        super(ResNetV1, self).__init__()
        if name == 'resnet50':
            self.l_to_ab = resnet50(in_channel=1, width=0.5)
            self.ab_to_l = resnet50(in_channel=2, width=0.5)
        elif name == 'resnet18':
            self.l_to_ab = resnet18(in_channel=1, width=0.5)
            self.ab_to_l = resnet18(in_channel=2, width=0.5)
        elif name == 'resnet101':
            self.l_to_ab = resnet101(in_channel=1, width=0.5)
            self.ab_to_l = resnet101(in_channel=2, width=0.5)
        else:
            raise NotImplementedError('model {} is not implemented'.format(name))

    def forward(self, x, layer=7):
        l, ab = torch.split(x, [1, 2], dim=1)
        feat_l = self.l_to_ab(l, layer)
        feat_ab = self.ab_to_l(ab, layer)
        return feat_l, feat_ab


class ResNetV2(nn.Module):
    def __init__(self, name='resnet50'):
        super(ResNetV2, self).__init__()
        if name == 'resnet50':
            self.l_to_ab = resnet50(in_channel=1, width=1)
            self.ab_to_l = resnet50(in_channel=2, width=1)
        elif name == 'resnet18':
            self.l_to_ab = resnet18(in_channel=1, width=1)
            self.ab_to_l = resnet18(in_channel=2, width=1)
        elif name == 'resnet101':
            self.l_to_ab = resnet101(in_channel=1, width=1)
            self.ab_to_l = resnet101(in_channel=2, width=1)
        else:
            raise NotImplementedError('model {} is not implemented'.format(name))

    def forward(self, x, layer=7):
        l, ab = torch.split(x, [1, 2], dim=1)
        feat_l = self.l_to_ab(l, layer)
        feat_ab = self.ab_to_l(ab, layer)
        return feat_l, feat_ab


class ResNetV3(nn.Module):
    def __init__(self, name='resnet50'):
        super(ResNetV3, self).__init__()
        if name == 'resnet50':
            self.l_to_ab = resnet50(in_channel=1, width=2)
            self.ab_to_l = resnet50(in_channel=2, width=2)
        elif name == 'resnet18':
            self.l_to_ab = resnet18(in_channel=1, width=2)
            self.ab_to_l = resnet18(in_channel=2, width=2)
        elif name == 'resnet101':
            self.l_to_ab = resnet101(in_channel=1, width=2)
            self.ab_to_l = resnet101(in_channel=2, width=2)
        else:
            raise NotImplementedError('model {} is not implemented'.format(name))

    def forward(self, x, layer=7):
        l, ab = torch.split(x, [1, 2], dim=1)
        feat_l = self.l_to_ab(l, layer)
        feat_ab = self.ab_to_l(ab, layer)
        return feat_l, feat_ab


class MyResNetsCMC(nn.Module):
    def __init__(self, name='resnet50v1'):
        super(MyResNetsCMC, self).__init__()
        if name.endswith('v1'):
            self.encoder = ResNetV1(name[:-2])
        elif name.endswith('v2'):
            self.encoder = ResNetV2(name[:-2])
        elif name.endswith('v3'):
            self.encoder = ResNetV3(name[:-2])
        else:
            raise NotImplementedError('model not support: {}'.format(name))

        self.encoder = nn.DataParallel(self.encoder)

    def forward(self, x, layer=7):
        return self.encoder(x, layer)
