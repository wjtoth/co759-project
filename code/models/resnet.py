# resnet on cifar-10 structure reproduced from ternarynet code at
# https://github.com/czhu95/ternarynet/blob/master/examples/Ternary-Net/tw-cifar10-resnet.py

import math
from copy import deepcopy
from collections import OrderedDict, Iterable
from itertools import repeat

import torch
import torch.nn as nn
from torch.autograd import Variable


class ResNet(nn.Module):
    def __init__(self, n, nonlin=nn.ReLU, use_bn=True):
        super(ResNet, self).__init__()
        self.n = n
        self.use_bn = use_bn

        if not use_bn:
            raise NotImplementedError('batch is currently required with resnet')

        def conv(in_chan, out_chan, stride=1):
            return nn.Conv2d(in_chan, out_chan, kernel_size=3, stride=stride, bias=not use_bn, padding=1)

        class ResidualBlock(nn.Module):
            def __init__(self, in_chan, increase_dim=False, is_first=False):
                super(ResidualBlock, self).__init__()
                res_block = OrderedDict()
                if increase_dim:
                    out_chan = in_chan * 2
                    stride1 = 2
                else:
                    out_chan = in_chan
                    stride1 = 1
                if not is_first:
                    res_block['batchnorm1'] = nn.BatchNorm2d(in_chan)
                    res_block['nonlin1'] = nonlin()
                res_block['conv1'] = conv(in_chan, out_chan, stride=stride1)
                res_block['batchnorm2'] = nn.BatchNorm2d(out_chan)
                res_block['nonlin2'] = nonlin()
                res_block['conv2'] = conv(out_chan, out_chan, stride=1)
                if increase_dim:
                    res_block['pool'] = nn.AvgPool2d(2)
#                    res_block['pad'] = nn.ZeroPad2d((0, in_chan // 2, 0, 0)) 
                    res_block['pad'] = PadChannels(in_chan // 2)
                self.res_block = nn.Sequential(res_block)

            def forward(self, x):
                x_in = x
                for m in self.res_block:
                    print(torch.typename(m), 'x:', x.size())
                    x = m(x)
                print('x:', x.size())
#                x = self.res_block(x)
                x = x_in + x
                return x

        layers = OrderedDict()

        layers['conv0'] = conv(3, 16, 1)
        layers['batchnorm0'] = nn.BatchNorm2d(16)
        layers['nonlin0'] = nonlin()

        layers['res1.0'] = ResidualBlock(16, is_first=True)
        for k in range(1, self.n):
            layers['res1.{}'.format(k)] = ResidualBlock(16)
        # 32,c=16

        layers['res2.0'] = ResidualBlock(16, increase_dim=True)
        for k in range(1, self.n):
            layers['res2.{}'.format(k)] = ResidualBlock(32)
        # 16,c=32

        layers['res3.0'] = ResidualBlock(32, increase_dim=True)
        for k in range(self.n):
            layers['res3.{}'.format(k)] = ResidualBlock(64)

        layers['batchnorm_last'] = nn.BatchNorm2d(64)
        layers['nonlin'] = nonlin()
        # 8,c=64

        layers['global_avg_pooling'] = nn.AvgPool2d(8)
        layers['fc1'] = nn.Linear(64, 10)

        self.layers = nn.Sequential(layers)

        # initialize weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        #x = self.layers(x)

        for m in self.layers:
            print(torch.typename(m), 'x:', x.size())
            x = m(x)
        print('x:', x.size())
        return x

    # def conv(name, l, channel, stride):
    #     return Conv2D(name, l, channel, 3, stride=stride,
    #                   nl=tf.identity, use_bias=False,
    #                   W_init=tf.random_normal_initializer(stddev=np.sqrt(2.0 / 9 / channel)))


class PadChannels(nn.Module):
    def __init__(self, padding):
        super(PadChannels, self).__init__()
        if not isinstance(padding, Iterable):
            padding = tuple(repeat(padding, 2))
        assert len(padding) == 1 or len(padding) == 2, 'padding must either be an int or a tuple of two ints'
        self.padding = padding

    def forward(self, x):
        assert len(x.size()) == 4
        print('pad input size:', x.size())
        n, c, h, w = x.size(0), x.size(1), x.size(2), x.size(3) 
        before_padding = Variable(x.data.new(n, self.padding[0], h, w).zero_())
        after_padding = Variable(x.data.new(n, self.padding[1], h, w).zero_())
        x = torch.cat((before_padding, x, after_padding), 1)
        print('pad output size:', x.size(), ' --', self.padding)
        return x
