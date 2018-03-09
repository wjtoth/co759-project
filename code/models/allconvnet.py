from torch import squeeze
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

class AllConvNet(nn.Module):
    def __init__(self, nonlin=F.relu, use_bn=False, nchan_in=3):
        super(AllConvNet, self).__init__()
        self.nonlin = nonlin
        if use_bn:
            raise NotImplementedError('batchnorm not supported for AllConvNet')
        # self.use_bn = use_bn
        self.conv1 = nn.Conv2d(nchan_in, 96, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(96, 96, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(96, 96, kernel_size=3, padding=1, stride=2)
        self.conv4 = nn.Conv2d(96, 192, kernel_size=3, padding=1)
        self.conv5 = nn.Conv2d(192, 192, kernel_size=3, padding=1)
        self.conv6 = nn.Conv2d(192, 192, kernel_size=3, padding=1, stride=2)
        self.conv7 = nn.Conv2d(192, 192, kernel_size=3, padding=1)
        self.conv8 = nn.Conv2d(192, 192, kernel_size=1, padding=0)
        self.conv9 = nn.Conv2d(192, 10, kernel_size=1, padding=0)
        # self.fc1 = nn.Linear(192*32*32, 10)
        relu_gain = init.calculate_gain('relu')
        init.xavier_normal(self.conv1.weight, gain=relu_gain)
        init.xavier_normal(self.conv2.weight, gain=relu_gain)
        init.xavier_normal(self.conv3.weight, gain=relu_gain)
        init.xavier_normal(self.conv4.weight, gain=relu_gain)
        init.xavier_normal(self.conv5.weight, gain=relu_gain)
        init.xavier_normal(self.conv6.weight, gain=relu_gain)
        init.xavier_normal(self.conv7.weight, gain=relu_gain)
        init.xavier_normal(self.conv8.weight, gain=relu_gain)
        init.xavier_normal(self.conv9.weight, gain=relu_gain)

    def forward(self, x):
        x = F.dropout(x, training=self.training, p=0.2)
        x = self.nonlin(self.conv1(x))
        x = self.nonlin(self.conv2(x))
        x = self.nonlin(self.conv3(x))
        x = F.dropout(x, training=self.training)
        x = self.nonlin(self.conv4(x))
        x = self.nonlin(self.conv5(x))
        x = self.nonlin(self.conv6(x))
        x = F.dropout(x, training=self.training)
        x = self.nonlin(self.conv7(x))
        x = self.nonlin(self.conv8(x))
        # x = self.fc1(x.view(-1, 32*32*192))
        x = self.nonlin(self.conv9(x))
        x = squeeze(F.avg_pool2d(x, kernel_size=x.size()[2:]))
        return x

    def conv_parameters(self):
        return self.parameters()

    def nonconv_parameters(self):
        return {}