# python
from collections import OrderedDict
from functools import partial

# pytorch
import torch
import torch.nn as nn
from torch.autograd import Function, Variable

# Friesen and Domingos
import targetprop as tp
from util.reshapemodule import ReshapeBatch
from activations import CAbs


class StepF(Function):
    """
    A step function that returns values in {-1, 1} and uses targetprop to
    update upstream weights in the network.
    """

    def __init__(self, targetprop_rule=tp.TPRule.WtHinge, make01=False, 
                 scale_by_grad_out=False, tanh_factor=1.0):
        super().__init__()
        self.tp_rule = targetprop_rule
        self.make01 = make01
        self.scale_by_grad_out = scale_by_grad_out
        self.target = None
        self.saved_grad_out = None
        self.tanh_factor = tanh_factor

    def forward(self, input_):
        self.save_for_backward(input_)
        output = tp.sign11(input_)  # output \in {-1, +1}
        if self.make01:
            output.clamp_(min=0)  # output \in {0, 1}
        return output

    def backward(self, grad_output):
        input_, = self.saved_tensors
        grad_input = None
        if self.needs_input_grad[0]:
            # compute targets = neg. sign of output grad, 
            # where t \in {-1, 0, 1} (t = 0 means ignore this unit)
            go = grad_output if self.saved_grad_out is None else self.saved_grad_out
            tp_grad_func = tp.TPRule.get_backward_func(self.tp_rule)
            if self.tp_rule.value == 27:
                tp_grad_func = partial(tp_grad_func, tanh_factor=self.tanh_factor)
            grad_input, self.target = tp_grad_func(input_, go, self.target, self.make01)
            if self.scale_by_grad_out:
                # remove batch-size scaling
                grad_input = grad_input * go.shape[0] * go.abs()  
        return grad_input


class Step(nn.Module):
    def __init__(self, targetprop_rule=tp.TPRule.TruncWtHinge, make01=False, 
                 scale_by_grad_out=False, tanh_factor=1.0):
        super().__init__()
        self.tp_rule = targetprop_rule
        self.make01 = make01
        self.scale_by_grad_out = scale_by_grad_out
        self.tanh_factor = tanh_factor
        self.output_hook = None

    def __repr__(self):
        s = "{name}(a={a}, b={b}, tp={tp})"
        a = 0 if self.make01 else -1
        return s.format(name=self.__class__.__name__, a=a, b=1, tp=self.tp_rule)

    def register_output_hook(self, output_hook):
        self.output_hook = output_hook

    def forward(self, x):
        function = StepF(targetprop_rule=self.tp_rule, make01=self.make01, 
                         scale_by_grad_out=self.scale_by_grad_out, 
                         tanh_factor=self.tanh_factor)
        y = function(x)
        if self.output_hook:
            # detach the output from the input to the next layer, 
            # so we can perform target propagation
            z = Variable(y.data.clone(), requires_grad=True)
            self.output_hook(x, y, z)
            return z
        else:
            return y


class ToyNet(nn.Module):

    def __init__(self, nonlin=nn.ReLU, input_shape=(784,), hidden_units=100,
                 num_classes=10, biases=True, separate_activations=True, 
                 multi_gpu_modules=False):
        super().__init__()
        self.input_size = input_shape[0]
        self.fc1_size = hidden_units
        self.separate_activations = separate_activations

        self.input_sizes = [list(input_shape), [self.fc1_size]]

        block1 = OrderedDict([
            ("fc1", nn.Linear(self.input_size, self.fc1_size, bias=biases)), 
            ("nonlin1", nonlin()),
        ])
        block2 = OrderedDict([
            ("fc2", nn.Linear(self.fc1_size, num_classes)),
        ])

        if self.separate_activations:
            del block1["nonlin1"]

        block1 = nn.Sequential(block1)
        block2 = nn.Sequential(block2)
        if multi_gpu_modules:
            block1, block2 = nn.DataParallel(block1), nn.DataParallel(block2)

        if separate_activations:
            self.all_modules = nn.ModuleList([block1, block2])
            self.all_activations = nn.ModuleList([nonlin(),])
        else:
            self.all_modules = nn.Sequential(OrderedDict([
                ("block1", block1), ("block2", block2),
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


class ConvNet4(nn.Module):

    def __init__(self, nonlin=nn.ReLU, use_batchnorm=False, input_shape=(3, 32, 32), 
                 num_classes=10, separate_activations=True, multi_gpu_modules=False):
        super().__init__()
        self.use_batchnorm = use_batchnorm
        self.conv1_size = 32
        self.conv2_size = 64
        self.fc1_size = 1024
        self.fc2_size = num_classes
        self.separate_activations = separate_activations

        if input_shape == (1, 28, 28):
            self.input_sizes = [
                list(input_shape), 
                [self.conv1_size, (input_shape[1]//4 + 1)*(input_shape[2]//4 + 1)//4 - 1, 
                 self.conv2_size//4 - 1], 
                [self.conv2_size, input_shape[1]//4, input_shape[2]//4], 
                [self.fc1_size],
            ]
        else:
            self.input_sizes = [
                list(input_shape), 
                [self.conv1_size, (input_shape[1]//4)*(input_shape[2]//4)//4 + 1, 
                 self.conv2_size//4 + 1], 
                [self.conv2_size, input_shape[1] // 4, input_shape[2] // 4], 
                [self.fc1_size],
            ]

        block1 = OrderedDict([
            ("conv1", nn.Conv2d(input_shape[0], self.conv1_size, 
                                kernel_size=5, padding=3)),
            ("maxpool1", nn.MaxPool2d(2)),
            ("nonlin1", nonlin()),
        ])

        block2 = OrderedDict([
            ("conv2", nn.Conv2d(self.conv1_size, self.conv2_size, 
                                kernel_size=5, padding=2)),
            ("maxpool2", nn.MaxPool2d(2)),
            ("nonlin2", nonlin()),
        ])

        block3 = OrderedDict([
            ("batchnorm1", nn.BatchNorm2d(self.conv2_size)),
            ("reshape1", ReshapeBatch(-1)),
            ("fc1", nn.Linear((input_shape[1] // 4) * (input_shape[2] // 4) 
                              * self.conv2_size, self.fc1_size)),
            ("nonlin3", nonlin()),
        ])

        block4 = OrderedDict([
            ("batchnorm2", nn.BatchNorm1d(self.fc1_size)),
            ("fc2", nn.Linear(self.fc1_size, self.fc2_size))
        ])

        if not self.use_batchnorm:
            del block3["batchnorm1"]
            del block4["batchnorm2"]
        if self.separate_activations:
            del block1["nonlin1"]
            del block2["nonlin2"]
            del block3["nonlin3"]

        block1 = nn.Sequential(block1)
        block2 = nn.Sequential(block2)
        block3 = nn.Sequential(block3)
        block4 = nn.Sequential(block4)
        if multi_gpu_modules:
            block1 = nn.DataParallel(block1)
            block2 = nn.DataParallel(block2)
            block3 = nn.DataParallel(block3)
            block4 = nn.DataParallel(block4)

        if self.separate_activations:
            self.all_modules = nn.ModuleList([
                block1, block2, block3, block4,
            ])
            self.all_activations = nn.ModuleList([nonlin(), nonlin(), nonlin()])
        else:
            self.all_modules = nn.Sequential(OrderedDict([
                ("block1", block1), ("block2", block2),
                ("block3", block3), ("block4", block4),
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


class ConvNet8(nn.Module):
    def __init__(self, nonlin=nn.ReLU, use_batchnorm=True, use_dropout=True,
                 input_shape=(3, 40, 40), num_classes=10, no_step_last=False, 
                 separate_activations=True, multi_gpu_modules=False):
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
            raise NotImplementedError("No other input sizes are currently supported")

        self.input_sizes = [
                list(input_shape), 
                None,
                None,
                None,
                None,
                None,
                None,
        ]

        block0 = OrderedDict([
            # padding = valid
            ("conv0", nn.Conv2d(3, 48, kernel_size=5, padding=pad0, bias=True)),  
            ("maxpool0", nn.MaxPool2d(2)),  # padding = same
            ("nonlin1", nonlin())  # 18
        ])

        block1 = OrderedDict([
            # padding = same
            ("conv1", nn.Conv2d(48, 64, kernel_size=3, padding=1, bias=bias)),  
            ("batchnorm1", nn.BatchNorm2d(64, eps=1e-4)),
            ("nonlin1", nonlin()),
        ])

        block2 = OrderedDict([
            # padding = same
            ("conv2", nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=bias)),  
            ("batchnorm2", nn.BatchNorm2d(64, eps=1e-4)),
            ("maxpool1", nn.MaxPool2d(2)),      # padding = same
            ("nonlin2", nonlin()),  # 9
        ])

        block3 = OrderedDict([
            # padding = valid
            ("conv3", nn.Conv2d(64, 128, kernel_size=3, padding=0, bias=bias)),  
            ("batchnorm3", nn.BatchNorm2d(128, eps=1e-4)),
            ("nonlin3", nonlin()),  # 7
        ])

        block4 = OrderedDict([
            # padding = same
            ("conv4", nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=bias)),  
            ("batchnorm4", nn.BatchNorm2d(128, eps=1e-4)),
            ("nonlin4", nonlin()),
        ])

        block5 = OrderedDict([
            # padding = valid
            ("conv5", nn.Conv2d(128, 128, kernel_size=3, padding=0, bias=bias)),  
            ("batchnorm5", nn.BatchNorm2d(128, eps=1e-4)),
            ("nonlin5", nonlin()),  # 5
        ])

        block6 = OrderedDict([
            ("dropout", nn.Dropout2d()),
            # padding = valid
            ("conv6", nn.Conv2d(128, 512, kernel_size=ks6, padding=0, bias=bias)),  
            ("batchnorm6", nn.BatchNorm2d(512, eps=1e-4)),
            ("nonlin6", nonlin() if not no_step_last else CAbs()),
            # ("nonlin6", nonlin() if not relu_last_layer else nn.ReLU()),
        ])

        block7 = OrderedDict([
            ("reshape_fc1", ReshapeBatch(-1)),
            ("fc1", nn.Linear(512, num_classes, bias=True))
        ])

        if not self.use_batchnorm:
            del block1["batchnorm1"]
            del block2["batchnorm2"]
            del block3["batchnorm3"]
            del block4["batchnorm4"]
            del block5["batchnorm5"]
            del block6["batchnorm6"]
        if not self.use_dropout:
            del block6["dropout"]

        block0 = nn.Sequential(block0)
        block1 = nn.Sequential(block1)
        block2 = nn.Sequential(block2)
        block3 = nn.Sequential(block3)
        block4 = nn.Sequential(block4)
        block5 = nn.Sequential(block5)
        block6 = nn.Sequential(block6)
        block7 = nn.Sequential(block7)
        if multi_gpu_modules:
            block0 = nn.DataParallel(block0)
            block1 = nn.DataParallel(block1)
            block2 = nn.DataParallel(block2)
            block3 = nn.DataParallel(block3)
            block4 = nn.DataParallel(block4)
            block5 = nn.DataParallel(block5)
            block6 = nn.DataParallel(block6)
            block7 = nn.DataParallel(block7)

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
                ("block0", nn.Sequential(block0)),
                ("block1", nn.Sequential(block1)),
                ("block2", nn.Sequential(block2)),
                ("block3", nn.Sequential(block3)),
                ("block4", nn.Sequential(block4)),
                ("block5", nn.Sequential(block5)),
                ("block6", nn.Sequential(block6)),
                ("block7", nn.Sequential(block7)),
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