# python
import sys
import argparse
import numpy as np
from functools import partial
from time import time
from random import randint

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
from util.reshapemodule import ReshapeBatch

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
    parser.add_argument('--nonlin', type=str, default='relu',
                        choices=('step01', 'step11', 'relu'))
    parser.add_argument('--comb-opt', action='store_true', default=False,
                        help='if specified, combinatorial optimization methods ' 
                             'are used for target setting')
    parser.add_argument('--comb-opt-method', type=str, default='local_search',
                        choices=('local_search', 'genetic'))
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
                             'adversarial examples generated using adv-attack')
    parser.add_argument('--adv-attack', type=str, default='fgsm', 
                        choices=tuple(adversarial.ATTACKS.keys()))
    parser.add_argument('--adv-epsilon', type=float, default=0.25)
    parser.add_argument('--cuda', action='store_true', default=False,
                        help='if specified, use CPU only')
    parser.add_argument('--seed', type=int, default=468412397,
                        help='random seed')
    
    args = parser.parse_args()
    args.cuda = args.cuda and torch.cuda.is_available()
    
    train_loader, val_loader, _, num_classes = \
        create_datasets('cifar10', args.batch, args.test_batch, not args.no_aug, 
                        args.no_val, args.data_root, args.cuda, args.seed, 
                        args.nworkers, args.dbg_ds_size, args.download)
    if args.adv_eval:
        adversarial_eval_dataset = get_adversarial_dataset('cifar10', args)

    args.nonlin = args.nonlin.lower()
    if args.nonlin == 'relu':
        nonlin = nn.ReLU
    elif args.nonlin == 'step01':
        nonlin = partial(Step, make01=True)
    elif args.nonlin == 'step11':
        nonlin = partial(Step, make01=False)

    print('Creating network...')
    network = ConvNet4(nonlin=nonlin)
    network.needs_backward_twice = False
    if args.nonlin.startswith('step'):
        network.targetprop_rule = tp.TPRule.TruncWtHinge
        network.needs_backward_twice = tp.needs_backward_twice(tp.TPRule.TruncWtHinge)
    if args.cuda:
        network = network.cuda()
    
    criterion = partial(multiclass_hinge_loss, reduce_=False if args.comb_opt else True)
    optimizer = partial(optim.Adam, lr=0.001)

    if args.comb_opt:
        if args.comb_opt_method == 'local_search':
            target_optimizer = partial(
                LocalSearchOptimizer, batch_size=args.batch, 
                candidates=10, iterations=10)
        elif args.comb_opt_method == 'genetic':
            target_optimizer = partial(
                GeneticOptimizer, batch_size=args.batch, candidates=10, 
                parents=5, generations=10, populations=1)
        else:
            raise NotImplementedError
    else:
        target_optimizer = None

    train(network, train_loader, val_loader, criterion, optimizer, 
          args.epochs, target_optimizer=target_optimizer, use_gpu=args.cuda)
    print('Finished training')

    if args.adv_eval:
        print('Evaluating on adversarial examples...')
        failure_rate = evaluate_adversarially(
            network, adversarial_eval_dataset, 'untargeted_misclassify', 
            args.adv_attack, args.test_batch, num_classes, 
            args.adv_epsilon, args.cuda)
        print('Failure rate: %0.2f%%' % (100*failure_rate))


class TargetPropOptimizer:
    
    def __init__(self, modules, sizes, loss_functions, 
                 batch_size, state=[], use_gpu=True):
        self.modules = modules
        self.sizes = [[batch_size] + shape for shape in sizes]
        self.loss_functions = loss_functions
        self.state = list(state)
        self.use_gpu = use_gpu

    def generate_targets(self, train_step, module_index, input, 
                         label, target, base_targets=None):
        """
        Subclasses should override this method.
        Args:
            train_step: Int.
            module_index: Int.
            input: torch.Tensor; input to the model corresponding 
                to self.modules at this training step.
            label: torch.Tensor; label corresponding to input 
                at this training step.
            target: torch.Tensor; target tensor for module module_index 
                to be used for generating, evaluating, and choosing 
                the targets of module module_index-1.
            base_targets: List; set of target tensors to use for generating 
                new targets (optional, default: None).
        Should generate a target torch.Tensor, list of target torch.Tensors, 
        or dictionary of target torch.Tensors and store it in the optimizer state.
        """
        raise NotImplementedError

    def evaluate_targets(self, train_step, module_index, input, label, target):
        """
        Subclasses should override this method. The method should evaluate 
        each stored target and update the state of the optimizer 
        with the evaluation data. 
        """
        raise NotImplementedError

    def choose_targets(self, train_step, module_index, input, label, target):
        """
        Subclasses should override this method. The method should filter 
        the stored targets based on the evaluation data and return 
        a target torch.Tensor, list of target torch.Tensors, 
        or dictionary of target torch.Tensors.
        """
        raise NotImplementedError

    def step(self, train_step, module_index, target, input=None, 
             label=None, base_targets=None):
        self.generate_targets(train_step, module_index, input, 
                              label, target, base_targets)
        return self.choose_targets(train_step, module_index, 
                                   input, label, target)


def convert_to_1hot(target, noutputs, make_11=True):
    if target.dim() == 1:
        if target.is_cuda:
            target_1hot = torch.cuda.CharTensor(target.shape[0], noutputs)
        else:
            target_1hot = torch.CharTensor(target.shape[0], noutputs)
        indices = target.unsqueeze(1)
        target_1hot.zero_()
        target_1hot.scatter_(1, indices, 1)
    elif target.dim() == 2:
        if target.is_cuda:
            target_1hot = torch.cuda.CharTensor(
                target.shape[0], target.shape[1], noutputs)
        else:
            target_1hot = torch.CharTensor(
                target.shape[0], target.shape[1], noutputs)
        indices = target.unsqueeze(2)
        target_1hot.zero_()
        target_1hot.scatter_(2, indices, 1)
    else:
        raise ValueError("Target must have 1 or 2 dimensions, " 
                         "but it has {} dimensions".format(target.dim()))
    target_1hot = target_1hot.type(target.type())
    if make_11:
        target_1hot = target_1hot * 2 - 1
    return target_1hot


def multiclass_hinge_loss(input_, target, reduce_=True):
    """Compute hinge loss: max(0, 1 - input * target)"""
    if input_.dim() == 2:
        noutputs = input_.shape[1]
    elif input_.dim() == 3:
        noutputs = input_.shape[2]
    else:
        raise ValueError("Input must have 2 or 3 dimensions, " 
                         "but it has {} dimensions".format(input_.dim()))
    target_1hot = convert_to_1hot(target.data, noutputs, make_11=True).float()
    if type(input_) is Variable:
        target_1hot = Variable(target_1hot)
    # max(0, 1-z*t)
    loss = (-target_1hot * input_.float() + 1.0).clamp(min=0).sum(dim=1)
    if reduce_:
        loss = loss.mean(dim=0)
    return loss


class LocalSearchOptimizer(TargetPropOptimizer):

    def __init__(self, modules, sizes, loss_functions, batch_size, candidates=10, 
                 iterations=10, searches=1, state=[], use_gpu=True):
        super().__init__(modules, sizes, loss_functions, batch_size, state, use_gpu)
        self.candidates = candidates
        self.iterations = iterations
        self.searches = searches

    @staticmethod
    def boxflip(candidate, y0, y1):
        candidate[:,y0:y1] = -candidate[:,y0:y1]     
        return candidate

    def generate_candidate(self, module_index, target):
        loss_function = self.loss_functions[module_index]
        module = self.modules[module_index]
        if module_index > 0:
            target = target.float()
            loss_function = torch.nn.MSELoss(size_average=False, reduce=False)

        # Generate a candidate target
        if self.use_gpu:
            candidate = torch.cuda.FloatTensor(*self.sizes[module_index])
        else:
            candidate = torch.FloatTensor(*self.sizes[module_index])
        candidate.random_(0,2)
        candidate.mul_(2)
        candidate.add_(-1)

        # Local search to find candidates
        for k in range(0, self.iterations):
            # Flip tensor in chunks of rows to reduce number of loss evaluations
            candidates = []
            for i in range(0, self.candidates):
                perturb_size = candidate.shape[1] // self.candidates
                candidate = self.boxflip(candidate, perturb_size*i, perturb_size*(i+1))
                candidates.append(candidate)
                candidate = self.boxflip(candidate, perturb_size*i, perturb_size*(i+1))
            candidate_losses = self.evaluate_targets(
                module, loss_function, target, candidates)
            candidate_index, loss = self.choose_target(candidate_losses)
            candidate = candidates[candidate_index.data[0]]

        if self.use_gpu:
            candidate_var = Variable(candidate).cuda()
        else:
            candidate_var = Variable(candidate)
        self.state["candidates"].append((candidate_var, loss))

    def generate_targets(self, train_step, module_index, 
                         input, label, target, base_targets=None):
        self.state = {"candidates": []}
        for i in range(0, self.searches):
            self.generate_candidate(module_index, target)
        index, loss = self.choose_target(
            [loss for candidate, loss in self.state["candidates"]])
        return self.state["candidates"][index][0], loss

    def evaluate_targets(self, module, loss_function, target, candidates):
        candidate_batch = torch.stack(candidates)
        target_batch = torch.stack([target]*len(candidates))
        candidate_batch = candidate_batch.view(
            candidate_batch.shape[0]*candidate_batch.shape[1], 
            *candidate_batch.shape[2:])
        target_batch = target_batch.view(
                target_batch.shape[0]*target_batch.shape[1], 
                *target_batch.shape[2:])
        if self.use_gpu:
            candidate_var = Variable(candidate_batch).cuda()
        else:
            candidate_var = Variable(candidate_batch)
        output = module(candidate_var)
        losses = loss_function(output, target_batch)
        losses = losses.view(len(candidates), int(np.prod(target.shape)))
        return losses.mean(dim=1)  # mean everything but candidate batch dim
    
    def choose_target(self, losses):
        if isinstance(losses, list):
            candidate_index, loss = min(
                enumerate(losses), key=lambda element: element[1].data[0])
        else:
            loss, candidate_index =  torch.min(losses, 0)
        return candidate_index, loss

    def step(self, train_step, module_index, target, input=None, 
             label=None, base_targets=None):
        return self.generate_targets(train_step, module_index, input, 
                                     label, target, base_targets)


class Target:

    def __init__(self, size, random=True, use_gpu=True):
        self.use_gpu = use_gpu
        self.size = size
        if self.use_gpu:
            self.values = torch.cuda.FloatTensor(*self.size)
        else:
            self.values = torch.FloatTensor(*self.size)
        if random:
            self.values.random_(0,2)
            self.values.mul_(2)
            self.values.add_(-1)
        else:
            self.values.zero_()
        
    def crossover(self, parent1, parent2):
        x0 = randint(1, self.size[0]-2)
        y0 = randint(1, self.size[1]-2)
           
        self.values[:x0,:y0] = parent1.values[:x0,:y0] 
        self.values[x0+1:,:y0] = parent1.values[x0+1:,:y0] 

        self.values[:x0,y0+1:] = parent2.values[:x0,y0+1:] 
        self.values[x0+1:,y0+1:] = parent2.values[x0+1:,y0+1:] 


class GeneticOptimizer(TargetPropOptimizer):

    def __init__(self, modules, sizes, loss_functions, batch_size, candidates=10, 
                 parents=5, generations=5, populations=1, state=[], use_gpu=True):
        super().__init__(modules, sizes, loss_functions, batch_size, state, use_gpu)
        self.candidates = candidates
        self.parents = parents
        self.generations = generations
        self.populations = populations

    def generate_candidate(self, module_index, target):
        loss_function = self.loss_functions[module_index]
        module = self.modules[module_index]
        candidate_size = self.sizes[module_index]
        if module_index > 0:
            target = target.float()
            loss_function = torch.nn.MSELoss(size_average=False, reduce=False)

        # generate a population of targets
        population = []
        for i in range(self.candidates):
            # generate a random candidate target
            candidate = Target(candidate_size, random=True, use_gpu=self.use_gpu)           
            population.append(candidate)
        candidate_losses = self.evaluate_targets(
                module, loss_function, target, 
                [target.values for target in population])
        population = self.filter_targets(population, candidate_losses)

        # main loop
        for i in range(self.generations):
            # generate children
            for j in range(self.candidates):
                # choose two random parents, and add crossover to population
                # some other condition for crossing two targets could be used here
                c1, c2 = None, None
                while c1 == c2:
                    c1 = randint(0, self.parents-1)
                    c2 = randint(0, self.parents-1)
                child_candidate = Target(
                    candidate_size, random=False, use_gpu=self.use_gpu)
                child_candidate.crossover(population[c1], population[c2])          
                population.append(child_candidate)
            candidate_losses = self.evaluate_targets(
                module, loss_function, target, 
                [target.values for target in population])
            population = self.filter_targets(population, candidate_losses)

        if self.use_gpu:
            candidate_var = Variable(population[0].values).cuda()
        else:
            candidate_var = Variable(population[0].values)
        self.state["candidates"].append((candidate_var, population[0].loss))

    def generate_targets(self, train_step, module_index, 
                         input, label, target, base_targets=None):
        self.state = {"candidates": []}
        for i in range(0, self.populations):
            self.generate_candidate(module_index, target)
        index, loss = self.choose_target(
            [loss for candidate, loss in self.state["candidates"]])
        return self.state["candidates"][index][0], loss

    def evaluate_targets(self, module, loss_function, target, candidates):
        candidate_batch = torch.stack(candidates)
        target_batch = torch.stack([target]*len(candidates))
        candidate_batch = candidate_batch.view(
            candidate_batch.shape[0]*candidate_batch.shape[1], 
            *candidate_batch.shape[2:])
        target_batch = target_batch.view(
                target_batch.shape[0]*target_batch.shape[1], 
                *target_batch.shape[2:])
        if self.use_gpu:
            candidate_var = Variable(candidate_batch).cuda()
        else:
            candidate_var = Variable(candidate_batch)
        output = module(candidate_var)
        losses = loss_function(output, target_batch)
        losses = losses.view(len(candidates), int(np.prod(target.shape)))
        return losses.mean(dim=1)  # mean everything but candidate batch dim

    def filter_targets(self, candidates, losses):
        for i in range(losses.shape[0]):
            candidates[i].loss = losses[i]
        candidates.sort(key=lambda target: target.loss.data[0], reverse=False)
        return candidates[0:self.parents]
    
    def choose_target(self, losses):
        if isinstance(losses, list):
            candidate_index, loss = min(
                enumerate(losses), key=lambda element: element[1].data[0])
        else:
            loss, candidate_index =  torch.min(losses, 0)
        return candidate_index, loss

    def step(self, train_step, module_index, target, input=None, 
             label=None, base_targets=None):
        return self.generate_targets(train_step, module_index, input, 
                                     label, target, base_targets)


def accuracy(prediction, target):
    return 100 * prediction.max(dim=1)[1].eq(target).float().mean().cpu()


def accuracy_topk(prediction, target, k=5):
    _, predictions_top5 = torch.topk(prediction, k, dim=1, largest=True)
    return 100 * (predictions_top5 == target.unsqueeze(1)).max(
        dim=1)[0].float().mean().cpu()


def train(model, train_dataset_loader, eval_dataset_loader, loss_function, 
          optimizer, epochs, target_optimizer=None, use_gpu=True):
    model.train()
    train_per_layer = target_optimizer is not None
    if train_per_layer:
        optimizers = [optimizer(module.parameters()) for module in model.all_modules]
        modules = list(zip(model.all_modules, optimizers))[::-1]  # in reverse order
        target_optimizer = target_optimizer(
            modules=list(model.all_modules)[::-1],
            sizes=model.input_sizes[::-1],
            loss_functions=[loss_function]*len(model.all_modules),
            use_gpu=use_gpu)
    else:
        optimizer = optimizer(model.parameters())
    for epoch in range(epochs):
        evaluate(model, eval_dataset_loader, loss_function, log=True, use_gpu=use_gpu)
        last_time = time()
        model.train()
        for i, (inputs, labels, _) in enumerate(train_dataset_loader):
            inputs, labels = Variable(inputs), Variable(labels)
            if use_gpu:
                inputs, labels = inputs.cuda(), labels.cuda()
            if i % 10 == 1:
                if train_per_layer:
                    outputs = model(inputs)
                    loss = loss_function(outputs, labels)
                    batch_accuracy = accuracy(outputs, labels)
                    batch_accuracy_top5 = accuracy_topk(outputs, labels, k=5)
                if i == 1:
                    steps = 1
                else:
                    steps = 10
                current_time = time() 
                print('training --- epoch: %d, batch: %d, loss: %.3f, acc: %.3f, '
                      'acc_top5: %.3f, steps/sec: %.2f' 
                      % (epoch+1, i+1, loss.data[0], steps/(current_time-last_time), 
                         batch_accuracy, batch_accuracy_top5))
                last_time = current_time
            train_step = epoch*len(train_dataset_loader) + i
            targets = labels
            if train_per_layer:
                for j, (module, optimizer) in enumerate(modules):
                    optimizer.zero_grad()
                    if j == len(modules)-1:
                        # no target generation at initial layer/module
                        outputs = module(inputs)
                        loss = torch.nn.MSELoss(
                            size_average=False)(outputs, targets.float())
                    else:
                        targets, loss = target_optimizer.step(
                            train_step, j, targets, label=labels)
                    loss.backward()
                    optimizer.step()
            else:
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = loss_function(outputs, targets)
                loss.backward(retain_graph=model.needs_backward_twice)
                if model.needs_backward_twice:
                    optimizer.zero_grad()
                    loss.backward()
                optimizer.step()


def evaluate(model, dataset_loader, loss_function, log=True, use_gpu=True):
    model.eval()
    total_loss, total_accuracy, total_accuracy_top5 = 0, 0, 0
    for inputs, labels in dataset_loader:
        inputs = Variable(inputs, volatile=True)
        labels = Variable(labels, volatile=True)
        if use_gpu:
            inputs, labels = inputs.cuda(), labels.cuda()
        outputs = model(inputs)
        loss = loss_function(outputs, labels).data[0]
        total_loss += loss
        total_accuracy += accuracy(outputs, labels)
        total_accuracy_top5 += accuracy_topk(outputs, labels, k=5)

    loss = total_loss / len(dataset_loader)
    total_accuracy = total_accuracy / len(dataset_loader)
    total_accuracy_top5 = total_accuracy_top5 / len(dataset_loader)
    if log:
        print('\nevaluation --- loss: %.3f, acc: %.3f, acc_top5: %.3f \n' 
              % (loss, total_accuracy, total_accuracy_top5))
    return loss, total_accuracy, total_accuracy_top5


def get_adversarial_dataset(dataset_name, terminal_args, size=2000):
    args = terminal_args
    adv_eval_loader, _, _, _ = create_datasets(
        dataset_name, 1, 1, not args.no_aug, args.no_val, args.data_root, 
        args.cuda, args.seed, 1, args.dbg_ds_size, args.download)
    return [(image[0].numpy(), label.numpy()) 
            for image, label, _ in adv_eval_loader][:size]


def evaluate_adversarially(model, dataset, criterion, attack, 
                           batch_size, num_classes, epsilon, cuda=True):
    if batch_size == 0:
        batch_size = 1
    model.eval()
    adversarial_examples, foolbox_model = adversarial.generate_adversarial_examples(
        model, attack, dataset, criterion, pixel_bounds=(-255, 255), 
        num_classes=num_classes, epsilon=epsilon, cuda=cuda)
    failure_rate = adversarial.adversarial_eval(
        foolbox_model, adversarial_examples, criterion, batch_size=batch_size)
    return failure_rate


class ConvNet4(nn.Module):
    def __init__(self, nonlin=nn.ReLU, use_batchnorm=False, input_shape=(3, 32, 32)):
        super(ConvNet4, self).__init__()
        self.use_batchnorm = use_batchnorm
        self.conv1_size = 32
        self.conv2_size = 64
        self.fc1_size = 1024
        self.fc2_size = 10

        self.input_sizes = [
            list(input_shape), 
            [self.conv1_size, (input_shape[1]//4)*(input_shape[2]//4)//4, self.conv2_size//4], 
            [(input_shape[1] // 4) * (input_shape[2] // 4) * self.conv2_size], 
            [self.fc1_size],
        ]

        block1 = OrderedDict([
            ('conv1', nn.Conv2d(input_shape[0], self.conv1_size, 
                                kernel_size=5, padding=2)),
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

        if not self.use_batchnorm:
            del block3['batchnorm1']
            del block4['batchnorm2']

        self.all_modules = nn.Sequential(OrderedDict([
            ('block1', nn.Sequential(block1)),
            ('block2', nn.Sequential(block2)),
            ('block3', nn.Sequential(block3)),
            ('block4', nn.Sequential(block4))
        ]))

    def forward(self, x):
        y = self.all_modules(x)
        return y


class StepF(Function):
    """
    A step function that returns values in {-1, 1} and uses targetprop to
    update upstream weights in the network.
    """

    def __init__(self, targetprop_rule=tp.TPRule.WtHinge, 
                 make01=False, scale_by_grad_out=False):
        super(StepF, self).__init__()
        self.tp_rule = targetprop_rule
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
            # where t \in {-1, 0, 1} (t = 0 means ignore this unit)
            go = grad_output if self.saved_grad_out is None else self.saved_grad_out
            tp_grad_func = tp.TPRule.get_backward_func(self.tp_rule)
            grad_input, self.target = tp_grad_func(input_, go, self.target, self.make01)
            if self.scale_by_grad_out:
                # remove batch-size scaling
                grad_input = grad_input * go.shape[0] * go.abs()  
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
            self.output_hook(x, y, z)
            return z
        else:
            return y


if __name__ == '__main__':
    main()