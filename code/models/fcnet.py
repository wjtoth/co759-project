from collections import OrderedDict
import torch.nn as nn
from util.reshapemodule import ReshapeBatch


class FCNet(nn.Module):
    def __init__(self, nonlin=nn.ReLU, input_shape=(3, 32, 32), filter_frac=1.0):
        super(FCNet, self).__init__()

        num_input_channels = input_shape[0] * input_shape[1] * input_shape[2]

        num_filters1 = round(784*filter_frac)
        num_filters2 = round(1024*filter_frac)

        block0 = OrderedDict([
            ('reshape', ReshapeBatch(-1)),
            ('fc0', nn.Linear(num_input_channels, num_filters1)),
            ('nonlin0', nonlin())
        ])

        block1 = OrderedDict([
            ('fc1', nn.Linear(num_filters1, num_filters2)),
            ('nonlin1', nonlin())
        ])

        block2 = OrderedDict([
            ('fc2', nn.Linear(num_filters2, num_filters2)),
            ('nonlin2', nonlin())
        ])

        block3 = OrderedDict([
            ('fc3', nn.Linear(num_filters2, num_filters2)),
            ('nonlin3', nonlin())
        ])

        block4 = OrderedDict([
            ('fc1', nn.Linear(num_filters2, 10)),
        ])

        self.all_modules = nn.Sequential(OrderedDict([
            ('block0', nn.Sequential(block0)),
            ('block1', nn.Sequential(block1)),
            ('block2', nn.Sequential(block2)),
            ('block3', nn.Sequential(block3)),
            ('block4', nn.Sequential(block4)),
        ]))

    def forward(self, x):
        x = self.all_modules(x)
        return x
