
#import adversarial

#python
import argparse
import numpy as np
from util.reshapemodule import ReshapeBatch

#friesen and Domingos
import losses
from datasets import create_datasets
from collections import OrderedDict
import targetprop as tp

#pytorch
import torch
import torch.nn as nn
from torch.autograd import Function, Variable
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.models as models
import torchvision.transforms as transforms


def main():
    # argument definitions
    parser = argparse.ArgumentParser()

    parser.add_argument('--batch', type=int, default=64,
                        help='batch size to use for training')
    parser.add_argument('--test-batch', type=int, default=0,
                        help='batch size to use for validation and testing')
    parser.add_argument('--data-root', type=str, default='',
                        help='root directory for imagenet dataset (with separate train, val, test folders)')     
    parser.add_argument('--no-aug', action='store_true',
                        help='if specified, do not use data augmentation (default=True for MNIST, False for CIFAR10)')
    parser.add_argument('--download', action='store_true',
                        help='allow downloading of the dataset (not including imagenet) if not found')
    parser.add_argument('--dbg-ds-size', type=int, default=0,
                        help='debug: artificially limit the size of the training data')
    parser.add_argument('--nworkers', type=int, default=2,
                        help='number of workers to use for loading data from disk')
    parser.add_argument('--no-val', action='store_true', default=False,
                        help='if specified, do not create a validation set from the training data and '
                             'use it to choose the best model')
    #changed Default to True (Daniel)
    parser.add_argument('--no-cuda', action='store_true', default=True,
                        help='if specified, use CPU only')
    parser.add_argument('--seed', type=int, default=468412397,
                        help='random seed')
    
    args = parser.parse_args()    
    
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    
    train_loader, val_loader, test_loader, num_classes = \
        create_datasets('cifar10', args.batch, args.test_batch, not args.no_aug, args.no_val, args.data_root,
                        args.cuda, args.seed, args.nworkers, args.dbg_ds_size, args.download)
            #(args.ds, args.batch, args.test_batch, not args.no_aug, args.no_val, args.data_root, 
            #            args.cuda, args.seed, args.nworkers, args.dbg_ds_size, args.download)

    print("Creating Network...")
    net = ConvNet4()
    print("Network Created")
    
    criterion = losses.multiclass_hinge_loss
    optimizer = optim.Adam(net.parameters(), lr=0.001)
    
    for epoch in range(2):  # loop over the dataset multiple times
        
        print("Starting training epoch %d..."%(epoch+1))
        
        for i, (inputs, labels, index) in enumerate(train_loader):
            
            #Friesen named labels as targets, changed to avoid confusion (Daniel)
            # wrap them in Variable
            inputs, labels = Variable(inputs), Variable(labels)
    
            # zero the parameter gradients
            optimizer.zero_grad()
    
            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
    
            # print statistics
            if i % 10 == 0:    # print every 10 mini-batches
                loss = loss.data[0]
                print('Epoch: %d, Batch: %5d, Loss: %.3f'%(epoch + 1, i + 1, loss))

    print('Finished Training')
    
class ConvNet4(nn.Module):
    def __init__(self, nonlin=nn.ReLU, use_bn=False, input_shape=(3, 32, 32)):
        super(ConvNet4, self).__init__()
        # self.nonlin = nonlin
        self.use_bn = use_bn
        self.conv1_size = 32  # 64 #32
        self.conv2_size = 64  # 128 #64
        self.fc1_size = 1024  # 200 #500 #1024
        self.fc2_size = 10  # 1024 #200 #500 #1024

        block1 = OrderedDict([
            ('conv1', nn.Conv2d(input_shape[0], self.conv1_size, kernel_size=5, padding=3)),
            ('maxpool1', nn.MaxPool2d(2)),
            ('nonlin1', nonlin())
        ])

        block2 = OrderedDict([
            ('conv2', nn.Conv2d(self.conv1_size, self.conv2_size, kernel_size=5, padding=2)),
            ('maxpool2', nn.MaxPool2d(2)),
            ('nonlin2', nonlin()),
        ])

        block3 = OrderedDict([
            ('batchnorm1', nn.BatchNorm2d(self.conv2_size)),
            ('reshape1', ReshapeBatch(-1)),
            ('fc1', nn.Linear((input_shape[1] // 4) * (input_shape[2] // 4) * self.conv2_size, self.fc1_size)),
            ('nonlin3', nonlin()),
        ])

        block4 = OrderedDict([
            ('batchnorm2', nn.BatchNorm1d(self.fc1_size)),
            ('fc2', nn.Linear(self.fc1_size, self.fc2_size))
        ])

        if not self.use_bn:
            del block3['batchnorm1']
            del block4['batchnorm2']

        self.all_modules = nn.Sequential(OrderedDict([
            ('block1', nn.Sequential(block1)),
            ('block2', nn.Sequential(block2)),
            ('block3', nn.Sequential(block3)),
            ('block4', nn.Sequential(block4))
        ]))

    def forward(self, x):
        x = self.all_modules(x)
        return x

class StepF(Function):
    """
    A step function that returns values in {-1, 1} and uses targetprop to
    update upstream weights in the network.
    """

    def __init__(self, targetprop_rule=tp.TPRule.WtHinge, make01=False, scale_by_grad_out=False):
        super(StepF, self).__init__()
        self.tp_rule = targetprop_rule
        # self.tp_grad_func = tp.TPRule.get_backward_func(self.tp_rule)
        self.make01 = make01
        self.scale_by_grad_out = scale_by_grad_out
        self.target = None
        self.saved_grad_out = None
        assert not (self.tp_rule == tp.TPRule.SSTE and self.scale_by_grad_out), 'scale_by_grad and SSTE are incompatible'
        assert not (self.tp_rule == tp.TPRule.STE and self.scale_by_grad_out), 'scale_by_grad and STE are incompatible'
        assert not (self.tp_rule == tp.TPRule.Ramp and self.scale_by_grad_out), 'scale_by_grad and Ramp are incompatible'

    def forward(self, input_):
        self.save_for_backward(input_)
        # output = torch.sign(input_)  # output \in {-1, 0, +1}
        output = tp.sign11(input_)  # output \in {-1, +1}
        if self.make01:
            output.clamp_(min=0)  # output \in {0, 1}
        return output

    def backward(self, grad_output):
        input_, = self.saved_tensors
        grad_input = None
        if self.needs_input_grad[0]:
            # compute targets = neg. sign of output grad, where t \in {-1, 0, 1} (t=0 means ignore this unit)
            go = grad_output if self.saved_grad_out is None else self.saved_grad_out
            tp_grad_func = tp.TPRule.get_backward_func(self.tp_rule)
            grad_input, self.target = tp_grad_func(input_, go, self.target, self.make01)
            if self.scale_by_grad_out:
                grad_input = grad_input * go.size()[0] * go.abs()  # remove batch-size scaling
        return grad_input


class Step(nn.Module):
    def __init__(self, targetprop_rule=tp.TPRule.TruncWtHinge, make01=False, scale_by_grad_out=False):
        super(Step, self).__init__()
        self.tp_rule = targetprop_rule
        self.make01 = make01
        self.scale_by_grad_out = scale_by_grad_out
        self.output_hook = None

    def __repr__(self):
        s = '{name}(a={a}, b={b}, tp={tp})'
        a = 0 if self.make01 else -1
        return s.format(name=self.__class__.__name__, a=a, b=1, tp=self.tp_rule)

    def register_output_hook(self, output_hook):
        self.output_hook = output_hook

    def forward(self, x):
        y = StepF(targetprop_rule=self.tp_rule, make01=self.make01, scale_by_grad_out=self.scale_by_grad_out)(x)
        if self.output_hook:
            # detach the output from the input to the next layer, so we can perform target propagation
            z = Variable(y.data.clone(), requires_grad=True)
            # assert self.output_hook, "output hook must exist, otherwise output vars will be lost"
            self.output_hook(x, y, z)
            return z
        else:
            return y
    
#class BeamSearch(Optimizer):
    
if __name__ == '__main__':
    main()