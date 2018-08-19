# python
from collections import OrderedDict
from functools import partial

# pytorch
import torch
import torch.nn as nn
from torch.autograd import Function, Variable

# Friesen and Domingos
import targetprop as tp
from models.convnet8 import ConvNet8
from util.reshapemodule import ReshapeBatch


class StepF(Function):
    """
    A step function that returns values in {-1, 1} and uses targetprop to
    update upstream weights in the network.
    """

    def __init__(self, targetprop_rule=tp.TPRule.WtHinge, make01=False, 
                 scale_by_grad_out=False, tanh_factor=1.0, use_momentum=False, 
                 momentum_factor=0, momentum_state=None, batch_labels=None):
        super().__init__()
        self.tp_rule = targetprop_rule
        self.make01 = make01
        self.scale_by_grad_out = scale_by_grad_out
        self.target = None
        self.saved_grad_out = None
        self.tanh_factor = tanh_factor
        self.use_momentum = use_momentum
        self.momentum_factor = momentum_factor
        self.momentum_state = momentum_state
        self.batch_labels = batch_labels

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
            if self.use_momentum:
                momentum_tensor = self.momentum_state[
                    range(self.batch_labels.shape[0]), self.batch_labels]
                grad_input, self.target = tp_grad_func(
                    input_, go, None, self.make01, velocity=momentum_tensor.float(), 
                    momentum_factor=self.momentum_factor, return_target=False)
                self.momentum_state[range(self.batch_labels.shape[0]), 
                                    self.batch_labels] = self.target.long()
            else:
                grad_input, self.target = tp_grad_func(
                    input_, go, self.target, self.make01)
            if self.scale_by_grad_out:
                # remove batch-size scaling
                grad_input = grad_input * go.shape[0] * go.abs()  
        return grad_input


class Step(nn.Module):
    def __init__(self, targetprop_rule=tp.TPRule.TruncWtHinge, make01=False, 
                 scale_by_grad_out=False, tanh_factor=1.0, use_momentum=False, 
                 momentum_factor=0):
        super().__init__()
        self.tp_rule = targetprop_rule
        self.make01 = make01
        self.scale_by_grad_out = scale_by_grad_out
        self.tanh_factor = tanh_factor
        self.output_hook = None
        self.use_momentum = use_momentum
        self.momentum_factor = momentum_factor
        self.momentum_state = None
        self.batch_labels = None
        if use_momentum:
            self.function = StepF(targetprop_rule=self.tp_rule, make01=self.make01, 
                                  scale_by_grad_out=self.scale_by_grad_out, 
                                  tanh_factor=self.tanh_factor,
                                  use_momentum=self.use_momentum, 
                                  momentum_factor=self.momentum_factor, 
                                  momentum_state=self.momentum_state,
                                  batch_labels=self.batch_labels)

    def __repr__(self):
        s = "{name}(a={a}, b={b}, tp={tp})"
        a = 0 if self.make01 else -1
        return s.format(name=self.__class__.__name__, a=a, b=1, tp=self.tp_rule)

    def register_output_hook(self, output_hook):
        self.output_hook = output_hook

    def initialize_momentum_state(self, target_shape, num_classes):
        self.momentum_state = torch.zeros(
            target_shape[0], num_classes, *target_shape[1:]).type(torch.LongTensor)
        self.momentum_state = self.momentum_state.cuda()
        if self.use_momentum:
            self.function.momentum_state = self.momentum_state

    def forward(self, x):
        function = StepF(targetprop_rule=self.tp_rule, make01=self.make01, 
                         scale_by_grad_out=self.scale_by_grad_out, 
                         tanh_factor=self.tanh_factor,
                         use_momentum=self.use_momentum, 
                         momentum_factor=self.momentum_factor, 
                         batch_labels=self.batch_labels)
        if self.use_momentum:
            self.momentum_state = self.function.momentum_state
            self.function = function
            self.function.momentum_state = self.momentum_state
            y = self.function(x)
        else:
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

    def __init__(self, nonlin=nn.ReLU, input_shape=(784,), 
                 separate_activations=True, num_classes=10, multi_gpu_modules=False):
        super().__init__()
        self.input_size = input_shape[0]
        self.fc1_size = 100
        self.separate_activations = separate_activations

        self.input_sizes = [list(input_shape), [self.fc1_size]]

        block1 = OrderedDict([
            ("fc1", nn.Linear(self.input_size, self.fc1_size)), 
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
                 separate_activations=True, multi_gpu_modules=False):
        super().__init__()
        self.use_batchnorm = use_batchnorm
        self.conv1_size = 32
        self.conv2_size = 64
        self.fc1_size = 1024
        self.fc2_size = 10
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


