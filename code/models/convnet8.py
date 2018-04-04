# architecture replicated from DoReFaNet code at
# https://github.com/ppwwyyxx/tensorpack/blob/master/examples/DoReFa-Net

from collections import OrderedDict
import torch
import torch.nn as nn
from activations import CAbs
from util.reshapemodule import ReshapeBatch


class ConvNet8(nn.Module):
    def __init__(self, nonlin=nn.ReLU, use_batchnorm=True, use_dropout=True,
                 input_shape=(3, 40, 40), no_step_last=False, separate_activations=True):
        super().__init__()
        self.use_batchnorm = use_batchnorm
        self.use_dropout = use_dropout
        self.separate_activations = separate_activations
        bias = not use_batchnorm

        if input_shape[1] == 40:
            pad0 = 0
            ks6 = 5
        elif input_shape[1] == 32:
            pad0 = 2
            ks6 = 4
        else:
            raise NotImplementedError('no other input sizes are currently supported')

        block0 = OrderedDict([
            # padding = valid
            ('conv0', nn.Conv2d(3, 48, kernel_size=5, padding=pad0, bias=True)),  
            ('maxpool0', nn.MaxPool2d(2)),  # padding = same
            ('nonlin1', nonlin())  # 18
        ])

        block1 = OrderedDict([
            # padding = same
            ('conv1', nn.Conv2d(48, 64, kernel_size=3, padding=1, bias=bias)),  
            ('batchnorm1', nn.BatchNorm2d(64, eps=1e-4)),
            ('nonlin1', nonlin()),
        ])

        block2 = OrderedDict([
            # padding = same
            ('conv2', nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=bias)),  
            ('batchnorm2', nn.BatchNorm2d(64, eps=1e-4)),
            ('maxpool1', nn.MaxPool2d(2)),      # padding = same
            ('nonlin2', nonlin()),  # 9
        ])

        block3 = OrderedDict([
            # padding = valid
            ('conv3', nn.Conv2d(64, 128, kernel_size=3, padding=0, bias=bias)),  
            ('batchnorm3', nn.BatchNorm2d(128, eps=1e-4)),
            ('nonlin3', nonlin()),  # 7
        ])

        block4 = OrderedDict([
            # padding = same
            ('conv4', nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=bias)),  
            ('batchnorm4', nn.BatchNorm2d(128, eps=1e-4)),
            ('nonlin4', nonlin()),
        ])

        block5 = OrderedDict([
            # padding = valid
            ('conv5', nn.Conv2d(128, 128, kernel_size=3, padding=0, bias=bias)),  
            ('batchnorm5', nn.BatchNorm2d(128, eps=1e-4)),
            ('nonlin5', nonlin()),  # 5
        ])

        block6 = OrderedDict([
            ('dropout', nn.Dropout2d()),
            # padding = valid
            ('conv6', nn.Conv2d(128, 512, kernel_size=ks6, padding=0, bias=bias)),  
            ('batchnorm6', nn.BatchNorm2d(512, eps=1e-4)),
            ('nonlin6', nonlin() if not no_step_last else CAbs()),
            # ('nonlin6', nonlin() if not relu_last_layer else nn.ReLU()),
        ])

        block7 = OrderedDict([
            ('reshape_fc1', ReshapeBatch(-1)),
            ('fc1', nn.Linear(512, 10, bias=True))
        ])

        if not self.use_batchnorm:
            del block1['batchnorm1']
            del block2['batchnorm2']
            del block3['batchnorm3']
            del block4['batchnorm4']
            del block5['batchnorm5']
            del block6['batchnorm6']
        if not self.use_dropout:
            del block6['dropout']

        if self.separate_activations:
            self.all_modules = nn.ModuleList([
                nn.Sequential(block0),
                nn.Sequential(block1),
                nn.Sequential(block2),
                nn.Sequential(block3),
                nn.Sequential(block4),
                nn.Sequential(block5),
                nn.Sequential(block6),
                nn.Sequential(block7),
            ])
            self.all_activations = nn.ModuleList(
                [nonlin(), nonlin(), nonlin(), nonlin(), 
                 nonlin(), nonlin(), nonlin()])
        else:
            self.all_modules = nn.Sequential(OrderedDict([
                ('block0', nn.Sequential(block0)),
                ('block1', nn.Sequential(block1)),
                ('block2', nn.Sequential(block2)),
                ('block3', nn.Sequential(block3)),
                ('block4', nn.Sequential(block4)),
                ('block5', nn.Sequential(block5)),
                ('block6', nn.Sequential(block6)),
                ('block7', nn.Sequential(block7)),
            ]))

    def forward(self, x):
        if self.separate_activations:
            for i, module in enumerate(self.all_modules):
                if i == 0:
                    y = module(x)
                else:
                    y = module(y)
                if i != len(self.all_modules)-1:
                    y = self.all_activations[i](y)
        else:
            y = self.all_modules(x)
        return y
