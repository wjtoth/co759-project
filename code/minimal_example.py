# python
import os
import sys
import argparse
import random
import numpy as np
from functools import partial
from itertools import chain
from time import time, sleep
from collections import OrderedDict, Counter

# pytorch
import torch
import torch.nn as nn
from torch.autograd import Function, Variable
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.models as models
import torchvision.transforms as transforms

# Friesen and Domingos
import losses
import activations
import targetprop as tp
from datasets import create_datasets
from models.convnet8 import ConvNet8
from util.reshapemodule import ReshapeBatch
from util.tensorboardlogger import TensorBoardLogger

# ours
import adversarial
from graph_nn import get_graphs

def main():
    # argument definitions
    parser = argparse.ArgumentParser()

    # model arguments
    parser.add_argument('--model', type=str, default='convnet4',
                        choices=('convnet4', 'convnet8', 'toynet'))
    parser.add_argument('--nonlin', type=str, default='relu',
                        choices=('relu', 'step01', 'step11', 'staircase'))

    # training/optimization arguments
    parser.add_argument('--no-train', action='store_true', default=False)
    parser.add_argument('--batch', type=int, default=64,
                        help='batch size to use for training')
    parser.add_argument('--epochs', type=int, default=10,
                        help='number of epochs to train for')
    parser.add_argument('--test-batch', type=int, default=0,
                        help='batch size to use for validation and testing')
    parser.add_argument('--loss', type=str, default='cross_entropy',
                        choices=('cross_entropy', 'hinge'))
    parser.add_argument('--wtdecay', type=float, default=0)
    parser.add_argument('--lr-decay-factor', type=float, default=1.0,
                        help='factor by which to multiply the learning rate ' 
                             'at each value in <lr-decay-epochs>')
    parser.add_argument('--lr-decay-epochs', type=int, nargs='+', default=None,
                        help='list of epochs at which to multiply ' 
                             'the learning rate by <lr-decay>')

    # target propagation arguments
    parser.add_argument('--grad-tp-rule', type=str, default='SoftHinge',
                        choices=('WtHinge', 'TruncWtHinge', 'SoftHinge', 
                                 'STE', 'SSTE', 'SSTEAndTruncWtHinge'))
    parser.add_argument('--comb-opt', action='store_true', default=False,
                        help='if specified, combinatorial optimization methods ' 
                             'are used for target setting')
    parser.add_argument('--comb-opt-method', type=str, default='local_search',
                        choices=('local_search', 'genetic'))
    parser.add_argument('--target-momentum', action='store_true', default=False,
                        help='if specified, target momentum is used. '
                             'Note: only supported with gradient-based targetprop')
    parser.add_argument('--target-momentum-factor', type=float, default=0.0,
                        help='factor by which to multiply the momentum tensor '
                             'during target setting')

    # data arguments
    parser.add_argument('--dataset', type=str, default='cifar10',
                        choices=('mnist', 'cifar10', 'cifar100', 
                                 'svhn', 'imagenet', 'graphs'))
    parser.add_argument('--data-root', type=str, default='',
                        help='root directory for imagenet dataset '
                             '(with separate train, val, test folders)')     
    parser.add_argument('--no-aug', action='store_true', default=False,
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
                        help='if specified, do not create a validation set from '
                             'the training data and use it to choose the best model')

    # adversarial evaluation arguments
    parser.add_argument('--adv-eval', action='store_true', default=False, 
                        help='if specified, evaluates the network on ' 
                             'adversarial examples generated using adv-attack')
    parser.add_argument('--adv-attack', type=str, default='fgsm', 
                        choices=tuple(adversarial.ATTACKS.keys()) + ('all',))
    parser.add_argument('--adv-epsilon', type=float, default=0.25)

    # computation arguments
    parser.add_argument('--cuda', action='store_true', default=False,
                        help='if specified, use CPU only')
    parser.add_argument('--seed', type=int, default=468412397,
                        help='random seed')
    
    args = parser.parse_args()
    args.cuda = args.cuda and torch.cuda.is_available()
    args.grad_tp_rule = tp.TPRule[args.grad_tp_rule]
    args.lr_decay_epochs = [] if args.lr_decay_epochs is None else args.lr_decay_epochs
    if not args.no_aug:
        args.no_aug = True if args.dataset == 'mnist' else False
    
    if args.dataset == 'graphs':
        train_loader, val_loader = get_graphs(
            6, batch_size=args.batch, num_workers=args.nworkers)
        num_classes = 2
    else:
        train_loader, val_loader, _, num_classes = \
            create_datasets(args.dataset, args.batch, args.test_batch, not args.no_aug, 
                            args.no_val, args.data_root, args.cuda, args.seed, 
                            args.nworkers, args.dbg_ds_size, args.download)
    if args.adv_eval:
        adversarial_eval_dataset = get_adversarial_dataset(args)

    args.nonlin = args.nonlin.lower()
    if args.nonlin == 'relu':
        nonlin = nn.ReLU
    elif args.nonlin == 'step01':
        nonlin = partial(Step, make01=True, targetprop_rule=args.grad_tp_rule, 
                         use_momentum=args.target_momentum, 
                         momentum_factor=args.target_momentum_factor)
    elif args.nonlin == 'step11':
        nonlin = partial(Step, make01=False, targetprop_rule=args.grad_tp_rule,
                         use_momentum=args.target_momentum, 
                         momentum_factor=args.target_momentum_factor)
    elif args.nonlin == 'staircase':
        nonlin = partial(activations.Staircase, targetprop_rule=args.grad_tp_rule,
                         nsteps=5, margin=1, trunc_thresh=2)

    if args.dataset == 'mnist':
        input_shape = (1, 28, 28)
        if args.model == 'toynet':
            input_shape = (784,)
    elif args.dataset.startswith('cifar'):
        input_shape = (3, 32, 32)
    elif args.dataset == 'svhn':
        input_shape = (3, 40, 40)
    elif args.dataset == 'imagenet':
        input_shape = (3, 224, 224)
    elif args.dataset == 'graphs':
        input_shape = (15,)
    else:
        raise NotImplementedError('no other datasets currently supported')

    print('Creating network...')
    if args.model == 'convnet4':
        network = ConvNet4(nonlin=nonlin, input_shape=input_shape, 
                           separate_activations=args.comb_opt)
    elif args.model == 'convnet8':
        network = ConvNet8(nonlin=nonlin, input_shape=input_shape, 
                           separate_activations=args.comb_opt)
    elif args.model == 'toynet':
        if args.dataset not in ['mnist', 'graphs']:
            raise NotImplementedError(
                'Toy network can only be trained on MNIST or graph connectivity task.')
        network = ToyNet(nonlin=nonlin, input_shape=input_shape,
                         separate_activations=args.comb_opt, 
                         num_classes=num_classes)
    network.needs_backward_twice = False
    if args.nonlin.startswith('step') or args.nonlin == 'staircase':
        network.targetprop_rule = args.grad_tp_rule
        network.needs_backward_twice = tp.needs_backward_twice(args.grad_tp_rule)
    if args.cuda:
        network = network.cuda()

    tb_logger = TensorBoardLogger('new_logs')

    if args.no_train:
        print('Loading from last checkpoint...')
        checkpoint_state = load_checkpoint(
            os.path.join(os.curdir, 'new_logs'), args=args)
        model_state = checkpoint_state['model_state']
        network.load_state_dict(model_state)
    else:
        if args.loss == 'cross_entropy':
            criterion = torch.nn.CrossEntropyLoss(
                size_average=not args.comb_opt, reduce=not args.comb_opt)
        elif args.loss == 'hinge':
            criterion = partial(multiclass_hinge_loss, reduce_=not args.comb_opt)
        optimizer = partial(optim.Adam, lr=0.00025, weight_decay=args.wtdecay)
        if args.comb_opt:
            if args.nonlin != 'step11':
                raise NotImplementedError(
                    "Discrete opt methods currently only support nonlin = step11.")
            if args.comb_opt_method == 'local_search':
                target_optimizer = partial(
                    LocalSearchOptimizer, batch_size=args.batch, 
                    criterion="loss", perturb_size=1000, candidates=64, 
                    iterations=10, searches=1, search_type="beam")
            elif args.comb_opt_method == 'genetic':
                target_optimizer = partial(
                    GeneticOptimizer, batch_size=args.batch, candidates=10, 
                    parents=5, generations=10, populations=1)
            else:
                raise NotImplementedError
        else:
            target_optimizer = None

        train(network, train_loader, val_loader, criterion, optimizer, 
              args.epochs, num_classes, target_optimizer=target_optimizer, 
              logger=tb_logger, use_gpu=args.cuda, args=args, print_param_info=False)
        print('Finished training\n')

    if args.adv_eval:
        print('Evaluating on adversarial examples...')
        if args.adv_attack == 'all':
            for attack in adversarial.ATTACKS:
                failure_rate = evaluate_adversarially(
                    network, adversarial_eval_dataset, 'untargeted_misclassify', 
                    attack, args.test_batch, num_classes, 
                    args.adv_epsilon, args.cuda)
                print('Failure rate: %0.2f%%' % (100*failure_rate))
        else:
            failure_rate = evaluate_adversarially(
                network, adversarial_eval_dataset, 'untargeted_misclassify', 
                args.adv_attack, args.test_batch, num_classes, 
                args.adv_epsilon, args.cuda)
            print('Failure rate: %0.2f%%' % (100*failure_rate))


def convert_to_1hot(target, noutputs, make_11=True):
    if target.is_cuda:
        target_1hot = torch.cuda.CharTensor(target.shape[0], noutputs)
    else:
        target_1hot = torch.CharTensor(target.shape[0], noutputs)
    target_1hot.zero_()
    target_1hot.scatter_(1, target.unsqueeze(1), 1)
    target_1hot = target_1hot.type(target.type())
    if make_11:
        target_1hot = target_1hot * 2 - 1
    return target_1hot


def multiclass_hinge_loss(input_, target, size_average=True, reduce_=True):
    """Compute hinge loss: max(0, 1 - input * target)"""
    target_1hot = convert_to_1hot(target.data, input_.shape[1], make_11=True).float()
    if type(input_) is Variable:
        target_1hot = Variable(target_1hot)
    # max(0, 1-z*t)
    loss = (-target_1hot * input_.float() + 1.0).clamp(min=0).sum(dim=1)
    if reduce_:
        if size_average:
            loss = loss.mean(dim=0)
        else:
            loss = loss.sum(dim=0)
    return loss


def soft_hinge_loss(z, t, xscale=1.0, yscale=1.0, reduce_=True):
    loss = yscale * torch.tanh(-(z * t).float() * xscale) + 1
    if reduce_:
        loss = loss.sum(dim=1)
    return loss


def accuracy(prediction, target, average=True):
    if prediction.dim() == target.dim():
        accuracies = 100 * prediction.eq(target).float()
    else:
        accuracies = 100 * prediction.max(dim=1)[1].eq(target).float()
    if average:
        accuracy = accuracies.mean()
    else:
        accuracy = accuracies
    return accuracy


def accuracy_topk(prediction, target, k=5, average=True):
    _, predictions_topk = torch.topk(prediction, k, dim=1, largest=True)
    accuracy_topk = 100 * (predictions_topk 
        == target.unsqueeze(1)).max(dim=1)[0].float()
    if average:
        accuracy_topk = accuracy_topk.mean()
    return accuracy_topk


class Metrics(dict):

    def __getitem__(self, key):
        if isinstance(key, int):
            item = {key: values[key] for metric, values in self.items()}
        else:
            item = super().__getitem__(key)
        return item

    def append(self, metric_tuple):
        for metric, value in metric_tuple:
            self[metric].append(value)


class TargetPropOptimizer:
    
    def __init__(self, modules, sizes, activations, loss_functions, 
                 batch_size, state=[], criterion="loss", use_gpu=True):
        self.modules = modules
        self.sizes = [[batch_size] + shape for shape in sizes]
        self.activations = activations
        self.loss_functions = loss_functions
        self.state = {}
        self.criterion = criterion
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

    def evaluate_targets(self, module, module_index, loss_function, target, candidates):
        if torch.is_tensor(candidates):
            candidate_batch = candidates
        else:
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
        if module_index > 0:
            output = output.view_as(target_batch)
            if self.criterion != "loss":
                output = self.activations[module_index](output)
            if self.criterion == "accuracy_top5":
                raise RuntimeError("Can only use top 5 accuracy as "
                                   "optimization criteria at output layer.")
        if self.criterion == "loss":
            losses = loss_function(output, target_batch)
        elif self.criterion == "accuracy":
            losses = accuracy(output, target_batch, average=False)
        elif self.criterion == "accuracy_top5":
            losses = accuracy_topk(output, target_batch, average=False)
        losses = losses.view(len(candidates), int(np.prod(target.shape)))
        return losses.mean(dim=1)  # mean everything but candidate batch dim

    def step(self, train_step, module_index, target, input=None, 
             label=None, base_targets=None):
        return self.generate_targets(train_step, module_index, input, 
                                     label, target, base_targets)


def generate_neighborhood(base_tensor, masking_weights, size, radius):
    """
    Args:
        base_tensor: torch.LongTensor; base tensor whose 
            neighborhood is generated.
        masking_weights: torch.FloatTensor; weights for randomly sampling 
            indices to perturb. 
        size: Int; size of the neighborhood, that is, the number of 
            tensors to generate. 
        radius: Int; (expected) number of values to perturb in 
            generating each neighbor. 
    Returns:
        A total of 'size' randomly-generated neighbors of base_tensor. 
    """
    batch_mask = torch.stack([masking_weights]*size)
    batch_base = torch.stack([base_tensor]*size)
    sampling_prob = radius / base_tensor.numel()
    one_tensor = torch.ones(batch_base.shape, device=torch.device("cuda:0"))
    indices = torch.bernoulli(one_tensor*sampling_prob)
    sampling_mask = indices.float() * batch_mask
    indices = torch.bernoulli(sampling_mask)
    neighbourhood = batch_base * (indices*-2 + 1)
    return neighbourhood


def get_random_tensor(shape, make01=False, use_gpu=True):
    if use_gpu:
        candidate = torch.cuda.FloatTensor(*shape)
    else:
        candidate = torch.FloatTensor(*shape)
    candidate.random_(0,2)
    if not make01:
        candidate.mul_(2)
        candidate.add_(-1)
    return candidate


class LocalSearchOptimizer(TargetPropOptimizer):

    def __init__(self, *args, perturb_size=50, candidates=50, 
                 iterations=10, searches=1, search_type="beam", **kwargs):
        super().__init__(*args, **kwargs)
        self.perturb_size = perturb_size
        self.candidates = candidates
        self.iterations = iterations
        self.searches = searches
        self.search_type = search_type
        self.state = {"chosen_groups": []}

    def find_candidates(self, module_index, target, train_step, base_candidates=None):
        loss_function = self.loss_functions[module_index]
        module = self.modules[module_index]
        if module_index > 0:
            target = target.float()
            loss_function = partial(soft_hinge_loss, reduce_=False)

        candidates = base_candidates
        for i, [candidate, weight, perturb_size] in enumerate(candidates.copy()):
            candidates[i][1] = round(self.candidates*weight)
            if candidate is None:
                candidates[i][0] = get_random_tensor(
                    self.sizes[module_index], use_gpu=self.use_gpu)
            if perturb_size is None:
                candidates[i][2] = self.perturb_size
        candidates = [candidate_tuple for candidate_tuple in candidates 
                      if round(self.candidates*candidate_tuple[1])]

        # Local search to find better candidate
        masking_weights = torch.ones(
            candidates[0][0].shape, device=torch.device("cuda:0"))
        for i in range(self.iterations):
            neighbors = []
            for [candidate, nbhd_size, perturb_size] in candidates:
                if nbhd_size == 0:
                    raise RuntimeError("Neighborhood size should be nonzero!")
                else:
                    neighbors.append(candidate.unsqueeze(0))
                    neighbors.append(generate_neighborhood(
                        candidate, masking_weights, nbhd_size, perturb_size))
            nbhd_sizes = [size+1 for _, size, _ in candidates]
            candidate_batch = torch.cat(neighbors)
            candidate_losses = self.evaluate_targets(
                module, module_index, loss_function, target, candidate_batch)
            if self.search_type == "beam":
                candidate_losses = [candidate_losses]
                candidate_groups = [candidate_batch]
            else:
                candidate_losses = torch.split(candidate_losses, nbhd_sizes)
                candidate_groups = torch.split(candidate_batch, nbhd_sizes)
            chosen_candidates, chosen_losses = [], []
            for j, (losses, candidate_group) in enumerate(
                    zip(candidate_losses, candidate_groups)):
                k = len(candidates) if self.search_type == "beam" else 1
                candidate_indices, losses = self.choose_target(losses, k)
                best_candidates = candidate_group[candidate_indices]
                if self.search_type == "beam":
                    best_candidates = [
                        [candidate.squeeze(0), candidates[j][1], candidates[j][2]] 
                         for j, candidate in enumerate(best_candidates.chunk(k))]
                else:
                    best_candidates = [[best_candidates.squeeze(0), 
                                        candidates[j][1], candidates[j][2]]]
                chosen_candidates.extend(best_candidates)
                chosen_losses.extend(
                    [loss.squeeze(0) for loss in losses.chunk(k)] 
                    if self.search_type == "beam" else [losses.squeeze(0)])
            candidates = chosen_candidates

        for [candidate, _, _], loss in zip(candidates, chosen_losses):
            if self.use_gpu:
                candidate_var = Variable(candidate).cuda()
            else:
                candidate_var = Variable(candidate)
            self.state["candidates"].append((candidate_var.long(), loss))

    def generate_targets(self, train_step, module_index, 
                         input, label, target, base_targets=None):
        self.state["candidates"] = []
        for i in range(self.searches):
            self.find_candidates(module_index, target, train_step, base_targets)
        index, loss = self.choose_target(
            [loss for candidate, loss in self.state["candidates"]])
        if self.search_type == "parallel":
            self.state["chosen_groups"].append(index)
        return self.state["candidates"][index][0], loss
    
    def choose_target(self, losses, k=1):
        if isinstance(losses, list):
            if k != 1:
                raise NotImplementedError("Top-k element retrieval with k > 1 "
                                          "not implemented for lists.")
            candidate_indices, losses = min(
                enumerate(losses), key=lambda element: element[1].data[0])
        else:
            losses, candidate_indices = torch.topk(
                losses, k, dim=0, largest=False, sorted=False)
        return candidate_indices, losses


class NeighbourhoodIterator:
    
    def __init__(self, dimensions):
        self.dimensions = dimensions

    def next(self):
        """
        Should return a list of coordinates to flip for next point 
        in neighbourhood to explore. Returning [] indicates 
        there are no more neighbours to explore.
        """
        raise NotImplementedError


class RandomNeighbourhoodIterator(NeighbourhoodIterator):

    def __init__(self, dimension, samples):
        super().__init__(dimension)
        self.samples = samples
        self.x_size = self.dimensions[0]
        self.y_size = self.dimensions[1]
        self.sample_count = 0

    def next(self):
        x = random.randrange(self.x_size)
        y = random.randrange(self.y_size)
        self.sample_count += 1
        if self.sample_count <= self.samples:
            return [(x,y)]
        return []


class CompleteNeighbourhoodIterator(NeighbourhoodIterator):

    def __init__(self, dimensions):
        super().__init__(dimensions)
        self.x_size = self.dimensions[0]
        self.y_size = self.dimensions[1]
        self.x = -1
        self.y = self.y_size-1
        self.first_next = True

    def next(self):
        """
        Returns a list of coordinates to flip for next point 
        in neighbourhood to explore. Returning [] indicates 
        there are no more neighbours to explore.
        """
        self.y = (self.y + 1) % self.y_size
        if self.y == 0:
            self.x = (self.x + 1) % self.x_size
        if self.x == 0 and self.y == 0:
            if  not self.first_next:
                return []
            self.first_next = False
        return [(self.x, self.y)]


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
        x0 = random.randint(1, self.size[0]-2)
        y0 = random.randint(1, self.size[1]-2)
           
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
            loss_function = partial(soft_hinge_loss, reduce_=False)

        # generate a population of targets
        population = []
        for i in range(self.candidates):
            # generate a random candidate target
            candidate = Target(candidate_size, random=True, use_gpu=self.use_gpu)           
            population.append(candidate)
        candidate_losses = self.evaluate_targets(
                module, module_index, loss_function, target, 
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
                    c1 = random.randint(0, self.parents-1)
                    c2 = random.randint(0, self.parents-1)
                child_candidate = Target(
                    candidate_size, random=False, use_gpu=self.use_gpu)
                child_candidate.crossover(population[c1], population[c2])          
                population.append(child_candidate)
            candidate_losses = self.evaluate_targets(
                module, module_index, loss_function, target, 
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


def train(model, train_dataset_loader, eval_dataset_loader, loss_function, 
          optimizer, epochs, num_classes, target_optimizer=None, logger=None,
          use_gpu=True, args=None, print_param_info=False):
    model.train()
    batch_size = train_dataset_loader.batch_size
    batches = len(train_dataset_loader)
    train_per_layer = target_optimizer is not None
    if train_per_layer:
        optimizers = [optimizer(module.parameters()) for module in model.all_modules]
        lr_schedulers = [optim.lr_scheduler.MultiStepLR(optimizer, 
                            args.lr_decay_epochs, gamma=args.lr_decay_factor)
                         for optimizer in optimizers]
        modules = list(zip(model.all_modules, optimizers))[::-1]  # in reverse order
        target_optimizer = target_optimizer(
            modules=list(model.all_modules)[::-1],
            sizes=model.input_sizes[::-1],
            activations=list(model.all_activations[::-1]),
            loss_functions=[loss_function]*len(model.all_modules),
            use_gpu=use_gpu)
    else:
        optimizer = optimizer(model.parameters())
        lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, args.lr_decay_epochs, 
                                                      gamma=args.lr_decay_factor)
        lr_schedulers = [lr_scheduler]
        if args.target_momentum:
            activations = list(chain.from_iterable(
                [[child for i, child in enumerate(module) if i == len(module)-1] 
                 for module in model.all_modules[:-1]]))
            for i, activation in enumerate(activations):
                activation.initialize_momentum_state(
                    [batch_size] + model.input_sizes[i+1], num_classes)
    training_metrics = Metrics({'dataset_batches': batches,
                                'loss': [], 'accuracy': [], 
                                'accuracy_top5': [], 'steps/sec': []})
    eval_metrics = Metrics({'dataset_batches': len(eval_dataset_loader), 
                            'loss': [], 'accuracy': [], 'accuracy_top5': []})
    for epoch in range(epochs):
        for scheduler in lr_schedulers:
            scheduler.step()
        if epoch == 0:
            evaluate(model, eval_dataset_loader, loss_function, 
                     eval_metrics, logger, 0, log=True, use_gpu=use_gpu)
        last_time = time()
        model.train()
        for i, batch in enumerate(train_dataset_loader):
            if args.dataset == 'graphs':
                inputs, labels = batch
            else:
                inputs, labels, _ = batch
            if inputs.shape[0] != batch_size:
                continue
            inputs, labels = Variable(inputs), Variable(labels)
            if use_gpu:
                inputs, labels = inputs.cuda(), labels.cuda()
            if isinstance(model, ToyNet):
                try:
                    inputs = inputs.view(batch_size, 784)
                except RuntimeError:
                    inputs = inputs.view(batch_size, 15)
            train_step = epoch*batches + i
            if i % 10 == 1:
                last_time = eval_step(model, inputs, labels, loss_function, 
                                      training_metrics, logger, epoch, 
                                      i, train_step, last_time)
            targets = labels
            if train_per_layer:
                # Obtain activations / hidden states
                activations = []
                for j, (module, _) in enumerate(modules[::-1]):
                    if j == 0:
                        outputs = module(inputs)
                    else:
                        outputs = module(model.all_activations[j-1](outputs.detach()))
                    activations.append(outputs)
                activations.reverse()
                # Then target prop in reverse mode
                for j, (module, optimizer) in enumerate(modules):
                    optimizer.zero_grad()
                    outputs = activations[j]
                    if j == 0:
                        loss = loss_function(outputs, targets).mean()
                        output_loss = loss
                    else:
                        loss = soft_hinge_loss(outputs, targets.float()).mean()
                    loss.backward()
                    optimizer.step()
                    if j == 1 and i % 5 == 1:
                        # Check loss change after SGD update step
                        updated_loss = loss_function(model(inputs), labels).mean()
                        loss_delta = updated_loss - output_loss
                        logger.scalar_summary(
                            "train/loss_delta", loss_delta.item(), train_step)
                    if j != len(modules)-1:
                        activation = model.all_activations[len(modules)-1-j-1](
                            activations[j+1].detach())
                        targets, target_loss = target_optimizer.step(
                            train_step, j, targets, label=labels, 
                            base_targets=[[None, 1/4, 500] for i in range(4)])
                    if print_param_info and i % 100 == 1:
                        layer = len(modules)-1-j
                        for k, param in enumerate(module.parameters()):
                            if k == 0:
                                print('\nlayer {0} - weight matrices mean and variance: '
                                      '{1:.8f}, {2:.8f}\n'.format(
                                      layer, torch.mean(param.data), 
                                      torch.std(param.data)))
                                print('layer {0} - gradient mean and variance: '
                                      '{1:.8f}, {2:.8f}\n'.format(
                                      layer, torch.mean(param.grad.data), 
                                      torch.std(param.grad.data)))
            else:
                if args.target_momentum:
                    for activation in activations:
                        activation.batch_labels = targets
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = loss_function(outputs, targets)
                loss.backward(retain_graph=model.needs_backward_twice)
                if model.needs_backward_twice:
                    optimizer.zero_grad()
                    loss.backward()
                optimizer.step()
        evaluate(model, eval_dataset_loader, loss_function, 
                 eval_metrics, logger, train_step, log=True, use_gpu=use_gpu)
        store_checkpoint(model, optimizer, args, training_metrics, 
                         eval_metrics, epoch, os.path.join(os.curdir, 'new_logs'))
        if target_optimizer is not None:
            print(Counter(target_optimizer.state["chosen_groups"][-batches:]), "\n")


def eval_step(model, inputs, labels, loss_function, training_metrics, 
              logger, epoch, batch, step, last_step_time):
    model.eval()
    outputs = model(inputs)
    loss = loss_function(outputs, labels).mean().item()
    batch_accuracy = accuracy(outputs, labels).item()
    if isinstance(model, ToyNet):
        batch_accuracy_top5 = 0
    else:
        batch_accuracy_top5 = accuracy_topk(outputs, labels, k=5).item()
    if batch == 1:
        steps = 1
    else:
        steps = 10
    current_time = time()
    steps_per_sec = steps/(current_time-last_step_time)
    metric_tuple = (('loss', loss), ('accuracy', batch_accuracy), 
                    ('accuracy_top5', batch_accuracy_top5), 
                    ('steps/sec', steps_per_sec))
    training_metrics.append(metric_tuple)
    print('training --- epoch: %d, batch: %d, loss: %.3f, acc: %.3f, '
          'acc_top5: %.3f, steps/sec: %.2f' 
          % (epoch+1, batch+1, loss, batch_accuracy, 
             batch_accuracy_top5, steps_per_sec))
    if logger is not None:
        logger.scalar_summary('train/loss', loss, step)
        logger.scalar_summary('train/accuracy', batch_accuracy, step)
        logger.scalar_summary('train/top5_accuracy', batch_accuracy_top5, step)
    last_step_time = current_time
    model.train()
    return last_step_time


def evaluate(model, dataset_loader, loss_function, 
             eval_metrics, logger, step, log=True, use_gpu=True):
    model.eval()
    batch_size = dataset_loader.batch_size
    total_loss, total_accuracy, total_accuracy_top5 = 0, 0, 0
    for inputs, labels in dataset_loader:
        if inputs.shape[0] != batch_size:
            continue
        inputs = Variable(inputs, volatile=True)
        labels = Variable(labels, volatile=True)
        if use_gpu:
            inputs, labels = inputs.cuda(), labels.cuda()
        if isinstance(model, ToyNet):
            try:
                inputs = inputs.view(batch_size, 784)
            except RuntimeError:
                inputs = inputs.view(batch_size, 15)
        outputs = model(inputs)
        loss = loss_function(outputs, labels).mean().item()
        total_loss += loss
        total_accuracy += accuracy(outputs, labels).item()
        if not isinstance(model, ToyNet):
            total_accuracy_top5 += accuracy_topk(outputs, labels, k=5).item()

    loss = total_loss / len(dataset_loader)
    total_accuracy = total_accuracy / len(dataset_loader)
    if isinstance(model, ToyNet):
        total_accuracy_top5 = 0
    else:
        total_accuracy_top5 = total_accuracy_top5 / len(dataset_loader)
    if log:
        print('\nevaluation --- loss: %.3f, acc: %.3f, acc_top5: %.3f \n' 
              % (loss, total_accuracy, total_accuracy_top5))
        if logger is not None:
            logger.scalar_summary('eval/loss', loss, step)
            logger.scalar_summary('eval/accuracy', total_accuracy, step)
            logger.scalar_summary('eval/top5_accuracy', total_accuracy_top5, step)
    eval_metrics.append((('loss', loss), ('accuracy', total_accuracy), 
                         ('accuracy_top5', total_accuracy_top5)))

    return loss, total_accuracy, total_accuracy_top5


def get_adversarial_dataset(terminal_args, size=1000):
    args = terminal_args
    adv_eval_loader, _, _, _ = create_datasets(
        args.dataset, 1, 1, not args.no_aug, args.no_val, args.data_root, 
        args.cuda, args.seed, 0, args.dbg_ds_size, args.download)
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


def store_checkpoint(model, optimizer, terminal_args, training_metrics, 
                     eval_metrics, epoch, root_log_dir):
    dataset_dir = os.path.join(root_log_dir, terminal_args.dataset)
    log_dir = os.path.join(dataset_dir, terminal_args.model + '-' + terminal_args.nonlin 
        + '-bs' + str(terminal_args.batch) + '-' + terminal_args.loss)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    checkpoint_state = {
        'args': terminal_args,
        'model_state': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'training_metrics': training_metrics,
        'eval_metrics': eval_metrics,
        'epoch': epoch,
        'save_time': time()
    }
    file_name = 'model_checkpoint_epoch{}.state'.format(epoch)
    file_path = os.path.join(log_dir, file_name)
    torch.save(checkpoint_state, file_path)
    print('\nModel checkpoint saved at: ' 
          + '\\'.join(file_path.split('\\')[-2:]) + '\n')
    # delete old checkpoint
    if epoch > 9:
        previous = os.path.join(
            log_dir, 'model_checkpoint_epoch{}.state'.format(epoch-10))
        if os.path.exists(previous) and os.path.isfile(previous):
            os.remove(previous)


def load_checkpoint(root_log_dir, args=None, log_dir=None, epoch=None):
    if args is None and log_dir is None:
        raise ValueError('Need at least one argument to locate the checkpoint.')
    if args is not None:
        log_dir = os.path.join(os.path.join(root_log_dir, args.dataset), 
                               args.model + '-' + args.nonlin + '-bs' 
                               + str(args.batch) + '-' + args.loss)
    if epoch is None:
        checkpoint_files = [file_name for file_name in os.listdir(log_dir)
                            if file_name.startswith('model_checkpoint')]
        checkpoint_files.sort()
    else:
        checkpoint_files = [
            file_name for file_name in os.listdir(log_dir)
            if file_name.startswith('model_checkpoint_epoch{}'.format(epoch))
        ]

    checkpoint_state = torch.load(os.path.join(log_dir, checkpoint_files[-1]))
    return checkpoint_state


class ToyNet(nn.Module):

    def __init__(self, nonlin=nn.ReLU, input_shape=(784,), 
                 separate_activations=True, num_classes=10):
        super().__init__()
        self.input_size = input_shape[0]
        self.fc1_size = 100
        self.separate_activations = separate_activations

        self.input_sizes = [list(input_shape), [self.fc1_size]]

        block1 = OrderedDict([
            ('fc1', nn.Linear(self.input_size, self.fc1_size)), 
            ('nonlin1', nonlin()),
        ])
        block2 = OrderedDict([
            ('fc2', nn.Linear(self.fc1_size, num_classes)),
        ])

        if self.separate_activations:
            del block1['nonlin1']
            self.all_modules = nn.ModuleList(
                [nn.Sequential(block1), nn.Sequential(block2)])
            self.all_activations = nn.ModuleList([nonlin(),])
        else:
            self.all_modules = nn.Sequential(OrderedDict([
                ('block1', nn.Sequential(block1)),
                ('block2', nn.Sequential(block2)),
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

    def __init__(self, nonlin=nn.ReLU, use_batchnorm=False, 
                 input_shape=(3, 32, 32), separate_activations=True):
        super().__init__()
        self.use_batchnorm = use_batchnorm
        self.conv1_size = 32
        self.conv2_size = 64
        self.fc1_size = 1024
        self.fc2_size = 10
        self.separate_activations = separate_activations

        self.input_sizes = [
            list(input_shape), 
            [self.conv1_size, (input_shape[1]//4)*(input_shape[2]//4)//4 + 1, 
             self.conv2_size//4 + 1], 
            [self.conv2_size, input_shape[1] // 4, input_shape[2] // 4], 
            [self.fc1_size],
        ]

        block1 = OrderedDict([
            ('conv1', nn.Conv2d(input_shape[0], self.conv1_size, 
                                kernel_size=5, padding=3)),
            ('maxpool1', nn.MaxPool2d(2)),
            ('nonlin1', nonlin()),
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
        if self.separate_activations:
            del block1['nonlin1']
            del block2['nonlin2']
            del block3['nonlin3']

        if self.separate_activations:
            self.all_modules = nn.ModuleList([
                nn.Sequential(block1),
                nn.Sequential(block2),
                nn.Sequential(block3),
                nn.Sequential(block4),
            ])
            self.all_activations = nn.ModuleList([nonlin(), nonlin(), nonlin()])
        else:
            self.all_modules = nn.Sequential(OrderedDict([
                ('block1', nn.Sequential(block1)),
                ('block2', nn.Sequential(block2)),
                ('block3', nn.Sequential(block3)),
                ('block4', nn.Sequential(block4)),
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


class StepF(Function):
    """
    A step function that returns values in {-1, 1} and uses targetprop to
    update upstream weights in the network.
    """

    def __init__(self, targetprop_rule=tp.TPRule.WtHinge, make01=False, 
                 scale_by_grad_out=False, use_momentum=False, momentum_factor=0,
                 momentum_state=None, batch_labels=None):
        super(StepF, self).__init__()
        self.tp_rule = targetprop_rule
        self.make01 = make01
        self.scale_by_grad_out = scale_by_grad_out
        self.target = None
        self.saved_grad_out = None
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
            if self.use_momentum:
                momentum_tensor = self.momentum_state[
                    range(self.batch_labels.shape[0]), self.batch_labels]
                grad_input, self.target = tp_grad_func(
                    input_, go, None, self.make01, velocity=momentum_tensor.float(), 
                    momentum_factor=self.momentum_factor, return_target=True)
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
                 scale_by_grad_out=False, use_momentum=False, momentum_factor=0):
        super(Step, self).__init__()
        self.tp_rule = targetprop_rule
        self.make01 = make01
        self.scale_by_grad_out = scale_by_grad_out
        self.output_hook = None
        self.use_momentum = use_momentum
        self.momentum_factor = momentum_factor
        self.momentum_state = None
        self.batch_labels = None
        self.function = StepF(targetprop_rule=self.tp_rule, make01=self.make01, 
                              scale_by_grad_out=self.scale_by_grad_out, 
                              use_momentum=self.use_momentum, 
                              momentum_factor=self.momentum_factor, 
                              momentum_state=self.momentum_state,
                              batch_labels=self.batch_labels)

    def __repr__(self):
        s = '{name}(a={a}, b={b}, tp={tp})'
        a = 0 if self.make01 else -1
        return s.format(name=self.__class__.__name__, a=a, b=1, tp=self.tp_rule)

    def register_output_hook(self, output_hook):
        self.output_hook = output_hook

    def initialize_momentum_state(self, target_shape, num_classes):
        self.momentum_state = torch.zeros(
            target_shape[0], num_classes, *target_shape[1:]).type(torch.LongTensor)
        self.momentum_state = self.momentum_state.cuda()
        self.function.momentum_state = self.momentum_state

    def forward(self, x):
        self.momentum_state = self.function.momentum_state
        self.function = StepF(targetprop_rule=self.tp_rule, make01=self.make01, 
                              scale_by_grad_out=self.scale_by_grad_out, 
                              use_momentum=self.use_momentum, 
                              momentum_factor=self.momentum_factor, 
                              momentum_state=self.momentum_state,
                              batch_labels=self.batch_labels)
        y = self.function(x)
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
