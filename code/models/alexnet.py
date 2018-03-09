# adapted from the pytorch vision AlexNet implementation found at
# https://github.com/pytorch/vision/blob/master/torchvision/models/alexnet.py
# --> put all operations into separate blocks

from collections import OrderedDict

import torch.nn as nn
import torch.utils.model_zoo as model_zoo

from util.reshapemodule import ReshapeBatch

__all__ = ['AlexNet', 'alexnet']

model_urls = {
    'alexnet': 'https://download.pytorch.org/models/alexnet-owt-4df8aa71.pth',
}


class AlexNet(nn.Module):

    def __init__(self, nonlin=nn.ReLU, no_step_last=False, num_classes=1000):
        super(AlexNet, self).__init__()
        nl_name = 'nonlin'  # nonlin.__name__

        block0 = OrderedDict([
            ('conv1', nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2)),
            ('{}1'.format(nl_name), nonlin()),
        ])

        block1 = OrderedDict([
            ('maxpool1', nn.MaxPool2d(kernel_size=3, stride=2)),
            ('conv2', nn.Conv2d(64, 192, kernel_size=5, padding=2)),
            ('{}2'.format(nl_name), nonlin()),
        ])

        block2 = OrderedDict([
            ('maxpool2', nn.MaxPool2d(kernel_size=3, stride=2)),
            ('conv3', nn.Conv2d(192, 384, kernel_size=3, padding=1)),
            ('{}3'.format(nl_name), nonlin()),
        ])

        block3 = OrderedDict([
            ('conv4', nn.Conv2d(384, 256, kernel_size=3, padding=1)),
            ('{}4'.format(nl_name), nonlin()),
        ])

        block4 = OrderedDict([
            ('conv5', nn.Conv2d(256, 256, kernel_size=3, padding=1)),
            ('{}5'.format(nl_name), nonlin()),
        ])

        block5 = OrderedDict([
            ('maxpool3', nn.MaxPool2d(kernel_size=3, stride=2)),
            ('reshape1', ReshapeBatch(-1)),
            ('dropout1', nn.Dropout()),
            ('fc1', nn.Linear(256 * 6 * 6, 4096)),
            ('{}6'.format(nl_name), nonlin()),
        ])

        block6 = OrderedDict([
            ('dropout2', nn.Dropout()),
            ('fc2', nn.Linear(4096, 4096)),
            # ('{}7'.format(nl_name), nonlin()),
        ])

        if not no_step_last:
            block6['{}7'.format(nl_name)] = nonlin()
            block7 = OrderedDict([('fc3', nn.Linear(4096, num_classes))])
        else:
            block6['{}7'.format(nl_name)] = nn.ReLU()
            block6['fc3'] = nn.Linear(4096, num_classes)
            block7 = None

        layers_od = OrderedDict([
            ('block0', nn.Sequential(block0)),
            ('block1', nn.Sequential(block1)),
            ('block2', nn.Sequential(block2)),
            ('block3', nn.Sequential(block3)),
            ('block4', nn.Sequential(block4)),
            ('block5', nn.Sequential(block5)),
            ('block6', nn.Sequential(block6)),
        ])

        if block7 is not None:
            layers_od['block7'] = nn.Sequential(block7)

        self.layers = nn.Sequential(layers_od)

        # self.features = nn.Sequential(
        #     nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
        #     nonlin(inplace=True),
        #     nn.MaxPool2d(kernel_size=3, stride=2),
        #     nn.Conv2d(64, 192, kernel_size=5, padding=2),
        #     nonlin(inplace=True),
        #     nn.MaxPool2d(kernel_size=3, stride=2),
        #     nn.Conv2d(192, 384, kernel_size=3, padding=1),
        #     nonlin(inplace=True),
        #     nn.Conv2d(384, 256, kernel_size=3, padding=1),
        #     nonlin(inplace=True),
        #     nn.Conv2d(256, 256, kernel_size=3, padding=1),
        #     nonlin(inplace=True),
        #     nn.MaxPool2d(kernel_size=3, stride=2),
        # )
        # self.classifier = nn.Sequential(
        #     nn.Dropout(),
        #     nn.Linear(256 * 6 * 6, 4096),
        #     nonlin(inplace=True),
        #     nn.Dropout(),
        #     nn.Linear(4096, 4096),
        #     nonlin(inplace=True),
        #     nn.Linear(4096, num_classes),
        # )

    def forward(self, x):
        # x = self.features(x)
        # x = x.view(x.size(0), 256 * 6 * 6)
        # x = self.classifier(x)
        x = self.layers(x)
        return x


def alexnet(pretrained=False, nonlin=nn.ReLU, **kwargs):
    r"""AlexNet model architecture from the
    `"One weird trick..." <https://arxiv.org/abs/1404.5997>`_ paper.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        nonlin (Module name): The name of the Module to create for each non-linearity instance.
    """
    assert nonlin == nn.ReLU or not pretrained, 'pre-trained AlexNet only supports ReLU non-linearities'
    model = AlexNet(nonlin=nonlin, **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['alexnet']))
    return model
