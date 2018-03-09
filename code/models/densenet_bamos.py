# original densenet code from https://github.com/bamos/densenet.pytorch
# -- modified to support arbitrary non-linearities, remove batchnorm, replace 
#    avgpooling with maxpooling, and re-order pool(nonlin(x)) as nonlin(pool(x))

import torch

import torch.nn as nn
import torch.optim as optim

import torch.nn.functional as F
from torch.autograd import Variable

import torchvision.datasets as dset
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch.nn.init as wt_init

import torchvision.models as models

import sys
import math


class Bottleneck(nn.Module):
    def __init__(self, nChannels, growthRate, nonlin, useBN=True):
        super(Bottleneck, self).__init__()
        interChannels = 4*growthRate
        self.nonlin = nonlin
        self.useBN = useBN
        if self.useBN:
            self.bn1 = nn.BatchNorm2d(nChannels)
            self.bn2 = nn.BatchNorm2d(interChannels)
        self.conv1 = nn.Conv2d(nChannels, interChannels, kernel_size=1,
                               bias=not useBN)
        self.conv2 = nn.Conv2d(interChannels, growthRate, kernel_size=3,
                               padding=1, bias=not useBN)

    def forward(self, x):
        out = self.conv1(self.nonlin(self.bn1(x) if self.useBN else x))
        out = self.conv2(self.nonlin(self.bn2(out) if self.useBN else out))
        out = torch.cat((x, out), 1)
        return out


class SingleLayer(nn.Module):
    def __init__(self, nChannels, growthRate, nonlin, useBN=True):
        super(SingleLayer, self).__init__()
        self.nonlin = nonlin
        self.useBN = useBN
        if self.useBN:
            self.bn1 = nn.BatchNorm2d(nChannels)
        self.conv1 = nn.Conv2d(nChannels, growthRate, kernel_size=3,
                               padding=1, bias=not useBN)

    def forward(self, x):
        out = self.conv1(self.nonlin(self.bn1(x) if self.useBN else x))
        out = torch.cat((x, out), 1)
        return out


class Transition(nn.Module):
    def __init__(self, nChannels, nOutChannels, nonlin, useBN=True):
        super(Transition, self).__init__()
        self.nonlin = nonlin
        self.useBN = useBN
        if self.useBN:
            self.bn1 = nn.BatchNorm2d(nChannels)
        self.conv1 = nn.Conv2d(nChannels, nOutChannels, kernel_size=1,
                               bias=not useBN)

    def forward(self, x):
        out = self.conv1(self.nonlin(self.bn1(x) if self.useBN else x))
        # out = F.avg_pool2d(out, 2)
        out = F.max_pool2d(out, 2)
        return out


class DenseNet(nn.Module):
    def __init__(self, growthRate, depth, reduction, nClasses, bottleneck, nonlin, useBN=True):
        super(DenseNet, self).__init__()

        self.nonlin = nonlin
        self.useBN = useBN
        nDenseBlocks = (depth-4) // 3
        if bottleneck:
            nDenseBlocks //= 2

        nChannels = 2*growthRate
        self.conv1 = nn.Conv2d(3, nChannels, kernel_size=3, padding=1,
                               bias=False)
        self.dense1 = self._make_dense(nChannels, growthRate, nDenseBlocks, bottleneck)
        nChannels += nDenseBlocks*growthRate
        nOutChannels = int(math.floor(nChannels*reduction))
        self.trans1 = Transition(nChannels, nOutChannels, nonlin=self.nonlin, useBN=self.useBN)

        nChannels = nOutChannels
        self.dense2 = self._make_dense(nChannels, growthRate, nDenseBlocks, bottleneck)
        nChannels += nDenseBlocks*growthRate
        nOutChannels = int(math.floor(nChannels*reduction))
        self.trans2 = Transition(nChannels, nOutChannels, nonlin=self.nonlin, useBN=self.useBN)

        nChannels = nOutChannels
        self.dense3 = self._make_dense(nChannels, growthRate, nDenseBlocks, bottleneck)
        nChannels += nDenseBlocks*growthRate

        if self.useBN:
            self.bn1 = nn.BatchNorm2d(nChannels)
        self.fc = nn.Linear(nChannels, nClasses)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                if self.useBN:
                    n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                    m.weight.data.normal_(0, math.sqrt(2. / n))
                else:
                    wt_init.orthogonal(m.weight, math.sqrt(2.))  # TODO: remove this?
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()

    def _make_dense(self, nChannels, growthRate, nDenseBlocks, bottleneck):
        layers = []
        for i in range(int(nDenseBlocks)):
            if bottleneck:
                layers.append(Bottleneck(nChannels, growthRate, nonlin=self.nonlin, useBN=self.useBN))
            else:
                layers.append(SingleLayer(nChannels, growthRate, nonlin=self.nonlin, useBN=self.useBN))
            nChannels += growthRate
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.trans1(self.dense1(out))
        out = self.trans2(self.dense2(out))
        out = self.dense3(out)
        # out = torch.squeeze(F.avg_pool2d(self.nonlin(self.bn1(out) if self.useBN else out), 8))
        # out = torch.squeeze(self.nonlin(F.avg_pool2d(self.bn1(out) if self.useBN else out, 8)))
        out = torch.squeeze(self.nonlin(F.max_pool2d(self.bn1(out) if self.useBN else out, 8)))
        out = self.fc(out)
        # out = F.log_softmax(out)
        return out
