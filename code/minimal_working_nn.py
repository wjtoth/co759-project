# python
import sys
import argparse
import numpy as np
from util.reshapemodule import ReshapeBatch
from random import choice
from functools import partial

# pytorch
import torch
import torch.nn as nn
from torch.autograd import Function, Variable
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.models as models
import torchvision.transforms as transforms

# friesen and Domingos
import losses
from datasets import create_datasets
from collections import OrderedDict
import targetprop as tp

# ours
import adversarial


def main():
    # argument definitions
    parser = argparse.ArgumentParser()

    parser.add_argument('--batch', type=int, default=64,
                        help='batch size to use for training')
    parser.add_argument('--epochs', type=int, default=10,
                        help='number of epochs to train for')
    parser.add_argument('--test-batch', type=int, default=0,
                        help='batch size to use for validation and testing')
    parser.add_argument('--data-root', type=str, default='',
                        help='root directory for imagenet dataset '
                             '(with separate train, val, test folders)')     
    parser.add_argument('--no-aug', action='store_true',
                        help='if specified, do not use data augmentation ' 
                             '(default=True for MNIST, False for CIFAR10)')
    parser.add_argument('--download', action='store_true',
                        help='allow downloading of the dataset ' 
                             '(not including imagenet) if not found')
    parser.add_argument('--dbg-ds-size', type=int, default=0,
                        help='debug: artificially limit the size of the training data')
    parser.add_argument('--nworkers', type=int, default=2,
                        help='number of workers to use for loading data from disk')
    parser.add_argument('--no-val', action='store_true', default=False,
                        help='if specified, do not create a validation set '
                             'from the training data and use it to choose the best model')
    parser.add_argument('--adv-eval', action='store_true', default=False, 
                        help='if specified, evaluates the network on ' 
                             'adversarial examples generated using FGSM')
    parser.add_argument('--cuda', action='store_true', default=False,
                        help='if specified, use CPU only')
    parser.add_argument('--seed', type=int, default=468412397,
                        help='random seed')
    
    args = parser.parse_args()    
    
    args.cuda = args.cuda and torch.cuda.is_available()
    
    train_loader, val_loader, test_loader, num_classes = \
        create_datasets('cifar10', args.batch, args.test_batch, not args.no_aug, 
                        args.no_val, args.data_root, args.cuda, args.seed, 
                        args.nworkers, args.dbg_ds_size, args.download)
    if args.adv_eval:
        _, adv_eval_loader, _, num_classes = \
            create_datasets('cifar10', args.batch, 1, not args.no_aug, 
                            args.no_val, args.data_root, args.cuda, args.seed, 
                            args.nworkers, args.dbg_ds_size, args.download)
        adversarial_eval_dataset = [(image[0].numpy(), torch.LongTensor(label[0]).numpy()) 
                                    for image, label in adv_eval_loader]

    print("Creating Network...")
    
    #cifar
    input_shape=(3, 32, 32)
    
    #define nonlinear function to be used
    nonlin = Step
    
    net = [ConvNet4_1(nonlin=nonlin,input_shape=input_shape),
        ConvNet4_2(nonlin=nonlin,input_shape=input_shape),
        ConvNet4_3(nonlin=nonlin,input_shape=input_shape),
        ConvNet4_4(nonlin=nonlin,input_shape=input_shape)]
    
    if args.cuda:
        for i in range(4):
            net[i].cuda()
    print("Network Created")
    
    optimizer = [optim.Adam(net[i].parameters()) for i in range(4)]

    for epoch in range(args.epochs):  # loop over the dataset multiple times
        
        print("Starting training epoch %d..."%(epoch+1))
        
        for i, (inputs, labels, index) in enumerate(train_loader):
            
            # Friesen named labels as targets, changed to avoid confusion (Daniel)
            # wrap them in Variable
            inputs, labels = Variable(inputs), Variable(labels)
            if args.cuda:
                inputs, labels = inputs.cuda(), labels.cuda()
    
            # zero the parameter gradients
            for n in range(4):                
                optimizer[i].zero_grad()

            # forward and initialize the targets
            targets12 = net[0](inputs)
            targets23 = net[1](targets12)
            targets34 = net[2](targets23)
            
            targets =  [inputs.data,targets12.data,targets23.data,targets34.data,labels.data]
            
            #backward from layer 4 to 0 (4,3,2,1)
            for l in range(4,0,-1):
                if l == 4:                    
                    criterion = losses.multiclass_hinge_loss
                else:
                    criterion = partial(nn.functional.l1_loss,size_average=False)
                print('Epoch: %d, Batch: %5d, Layer %d->%d'%(epoch + 1, i + 1, l, l+1))  
                #optimize weights 
                bestloss = 0
                for j in range(100):
                    optimizer[l-1].zero_grad()
                    outputs = net[l-1](Variable(targets[l-1]))
                    loss = criterion(outputs, Variable(targets[l]))
                    loss.backward(retain_graph=True)
                    loss = loss.data[0]
                    print('Epoch: %d, Batch: %5d, Layer %d->%d, Loss: %.3f'%(epoch + 1, i + 1, l, l+1, loss))
                    optimizer[l-1].step()
                    bestloss = loss
                    if bestloss < 0.01:
                        break
                
                #local searching target settings, for layers 1-2, 2-3, 3-4, 4-output
                if l > 1:                    
                    for j,t in enumerate(targets[l-1]):
                        print('Epoch: %d, Batch: %5d, Local Searching Layer %d->%d,  \
                           Target %d,1, Loss: %.3f'%(epoch + 1, i + 1, l, l+1, j+1, bestloss))
                        for k,tt in enumerate(t):
                            #flip a target value
                            targets[l-1][j][k] = -targets[l-1][j][k]
                            
                            #run targets through weights and activations
                            outputs = net[l-1](Variable(targets[l-1]))
                            
                            #calculate loss
                            loss = criterion(outputs, Variable(targets[l]))
                            loss = loss.data[0]     
                            
                            #check if target setting improved loss
                            if loss < bestloss:
                                #update best solution
                                bestloss = loss
                            else:
                                #revert changes                        
                                targets[l-1][j][k] = -targets[l-1][j][k]   
                                
                            #break out for if targets are feasible enough
                            if bestloss < 0.01:
                                break
                        if bestloss < 0.01:
                            break    
    
    
    #need to implement a forward pass on all networks and evaluate the test set
    print('Finished training')

    if args.adv_eval:
        net.eval()
        print('Evaluating on adversarial examples...')
        adversarial_examples = adversarial.generate_adversarial_examples(
            net, "fgsm", adversarial_eval_dataset, "untargeted_misclassify", 
            pixel_bounds=(-255, 255), num_classes=num_classes)
        failure_rate = adversarial.adversarial_eval(net, adversarial_examples, 
            "untargeted_misclassify", batch_size=args.test_batch)
        print("Failure rate: %.2f\%" % failure_rate)
    
class ConvNet4(nn.Module):
    def __init__(self, nonlin=nn.ReLU, use_bn=False, input_shape=(3, 32, 32)):
        super(ConvNet4, self).__init__()
        # self.nonlin = nonlin
        self.use_bn = use_bn
        self.conv1_size = 32  # 64 #32
        self.conv2_size = 64  # 128 #64
        self.fc1_size = 1024  # 200 #500 #1024
        self.fc2_size = 10  # 1024 #200 #500 #1024
        
        #initialize the set of targets for the mid layers
        self.targets = [
                [choice([-1,1]) for i in range(self.conv1_size)],
                [choice([-1,1]) for i in range(self.conv2_size)],
                [choice([-1,1]) for i in range(self.fc1_size)]
                ]  # 1024 #200 #500 #1024
        
        block1 = OrderedDict([
            ('conv1', nn.Conv2d(input_shape[0], self.conv1_size, 
                                kernel_size=5, padding=3)),
            ('maxpool1', nn.MaxPool2d(2)),
            ('nonlin1', nonlin())
        ])

        block2 = OrderedDict([
            ('conv2', nn.Conv2d(self.conv1_size, self.conv2_size, 
                                kernel_size=5, padding=2)),
            ('maxpool2', nn.MaxPool2d(2)),
            ('nonlin2', nonlin()),
        ])

        block3 = OrderedDict([
            ('batchnorm1', nn.BatchNorm2d(self.conv2_size)),
            ('reshape1', ReshapeBatch(-1)),
            ('fc1', nn.Linear((input_shape[1] // 4) * (input_shape[2] // 4) 
                              * self.conv2_size, self.fc1_size)),
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


class ConvNet4_1(nn.Module):
    def __init__(self, nonlin=nn.ReLU, use_bn=False, input_shape=(3, 32, 32)):
        super(ConvNet4_1, self).__init__()
        # self.nonlin = nonlin
        self.conv1_size = 32  # 64 #32
        
        block1 = OrderedDict([
            ('conv1', nn.Conv2d(input_shape[0], self.conv1_size, 
                                kernel_size=5, padding=3)),
            ('maxpool1', nn.MaxPool2d(2)),
            ('nonlin1', nonlin())
        ])

        self.all_modules = nn.Sequential(OrderedDict([
            ('block1', nn.Sequential(block1))
        ]))

    def forward(self, x):
        x = self.all_modules(x)
        return x


class ConvNet4_2(nn.Module):
    def __init__(self, nonlin=nn.ReLU, use_bn=False, input_shape=(3, 32, 32)):
        super(ConvNet4_2, self).__init__()
        # self.nonlin = nonlin
        self.conv1_size = 32  # 64 #32
        self.conv2_size = 64  # 128 #64

        block2 = OrderedDict([
            ('conv2', nn.Conv2d(self.conv1_size, self.conv2_size, 
                                kernel_size=5, padding=2)),
            ('maxpool2', nn.MaxPool2d(2)),
            ('nonlin2', nonlin()),
        ])


        self.all_modules = nn.Sequential(OrderedDict([
            ('block2', nn.Sequential(block2))
        ]))

    def forward(self, x):
        x = self.all_modules(x)
        return x


class ConvNet4_3(nn.Module):
    def __init__(self, nonlin=nn.ReLU, use_bn=False, input_shape=(3, 32, 32)):
        super(ConvNet4_3, self).__init__()
        # self.nonlin = nonlin
        self.use_bn = use_bn
        self.conv2_size = 64  # 128 #64
        self.fc1_size = 1024  # 200 #500 #1024
        
        block3 = OrderedDict([
            ('batchnorm1', nn.BatchNorm2d(self.conv2_size)),
            ('reshape1', ReshapeBatch(-1)),
            ('fc1', nn.Linear((input_shape[1] // 4) * (input_shape[2] // 4) 
                              * self.conv2_size, self.fc1_size)),
            ('nonlin3', nonlin()),
        ])

        if not self.use_bn:
            del block3['batchnorm1']

        self.all_modules = nn.Sequential(OrderedDict([
            ('block3', nn.Sequential(block3))
        ]))

    def forward(self, x):
        x = self.all_modules(x)
        return x


class ConvNet4_4(nn.Module):
    def __init__(self, nonlin=nn.ReLU, use_bn=False, input_shape=(3, 32, 32)):
        super(ConvNet4_4, self).__init__()
        # self.nonlin = nonlin
        self.use_bn = use_bn
        self.fc1_size = 1024  # 200 #500 #1024
        self.fc2_size = 10  # 1024 #200 #500 #1024
        
        block4 = OrderedDict([
            ('batchnorm2', nn.BatchNorm1d(self.fc1_size)),
            ('fc2', nn.Linear(self.fc1_size, self.fc2_size))
        ])

        if not self.use_bn:
            del block4['batchnorm2']

        self.all_modules = nn.Sequential(OrderedDict([
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
    def __init__(self, targetprop_rule=tp.TPRule.WtHinge, 
                 make01=False, scale_by_grad_out=False):
        super(StepF, self).__init__()
        self.tp_rule = targetprop_rule
        # self.tp_grad_func = tp.TPRule.get_backward_func(self.tp_rule)
        self.make01 = make01
        self.scale_by_grad_out = scale_by_grad_out
        self.target = None
        self.saved_grad_out = None
        # assert not (self.tp_rule == tp.TPRule.SSTE and self.scale_by_grad_out), \
        #     'scale_by_grad and SSTE are incompatible'
        # assert not (self.tp_rule == tp.TPRule.STE and self.scale_by_grad_out), \
        #     'scale_by_grad and STE are incompatible'
        # assert not (self.tp_rule == tp.TPRule.Ramp and self.scale_by_grad_out), \
        #     'scale_by_grad and Ramp are incompatible'

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
            # where t \in {-1, 0, 1} (t=0 means ignore this unit)
            go = grad_output if self.saved_grad_out is None else self.saved_grad_out
            tp_grad_func = tp.TPRule.get_backward_func(self.tp_rule)
            grad_input, self.target = tp_grad_func(input_, go, self.target, self.make01)
            if self.scale_by_grad_out:
                # remove batch-size scaling
                grad_input = grad_input * go.size()[0] * go.abs()  
        return grad_input


class Step(nn.Module):
    def __init__(self, targetprop_rule=tp.TPRule.TruncWtHinge, 
                 make01=False, scale_by_grad_out=False):
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
        y = StepF(targetprop_rule=self.tp_rule, make01=self.make01, 
                  scale_by_grad_out=self.scale_by_grad_out)(x)
        if self.output_hook:
            # detach the output from the input to the next layer, 
            # so we can perform target propagation
            z = Variable(y.data.clone(), requires_grad=True)
            # assert self.output_hook, "output hook must exist, 
            # otherwise output vars will be lost"
            self.output_hook(x, y, z)
            return z
        else:
            return y
    
if __name__ == '__main__':
    main()