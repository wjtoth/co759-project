# python
import os
import re
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
    parser.add_argument('--collect-params', action='store_true', default=False,
                        help='if specified, collects model data '
                             'at various training steps for AMPL optimization; '
                             'currently only available for ToyNet model.')

    # target propagation arguments
    parser.add_argument('--grad-tp-rule', type=str, default='SoftHinge',
                        choices=('WtHinge', 'TruncWtHinge', 'SoftHinge', 
                                 'STE', 'SSTE', 'SSTEAndTruncWtHinge'))
    parser.add_argument('--comb-opt', action='store_true', default=False,
                        help='if specified, combinatorial optimization methods ' 
                             'are used for target setting')
    parser.add_argument('--comb-opt-method', type=str, default='local_search',
                        choices=('local_search', 'genetic', 'rand_grad'))
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
    parser.add_argument('--multi-gpu', action='store_true', default=False,
                        help='if specified, use all available GPUs')
    parser.add_argument('--seed', type=int, default=468412397,
                        help='random seed')

    # logging arguments
    parser.add_argument('--tb-logging', action='store_true', default=False,
                        help='if specified, enable logging Tensorboard summaries')
    parser.add_argument('--store-checkpoints', action='store_true', default=False, 
                        help='if specified, enables storage of the current model ' 
                             'and training parameters at each epoch')
    
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
        if args.model == 'toynet':
            input_shape = (3072,)
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
        if args.dataset not in ['cifar10', 'mnist', 'graphs']:
            raise NotImplementedError(
                'Toy network can only be trained on CIFAR10, '
                'MNIST, or graph connectivity task.')
        network = ToyNet(nonlin=nonlin, input_shape=input_shape,
                         separate_activations=args.comb_opt, 
                         num_classes=num_classes)
    network.needs_backward_twice = False
    if args.nonlin.startswith('step') or args.nonlin == 'staircase':
        network.targetprop_rule = args.grad_tp_rule
        network.needs_backward_twice = tp.needs_backward_twice(args.grad_tp_rule)
    if args.cuda:
        network = network.cuda()
    if args.multi_gpu:
        network = nn.DataParallel(
            network, device_ids=list(range(torch.cuda.device_count())))

    tb_logger = TensorBoardLogger('new_logs') if args.tb_logging else None

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
                    SearchOptimizer, batch_size=args.batch, 
                    criterion="loss", regions=10, 
                    perturb_scheduler=lambda x, y, z: 1000, 
                    candidates=64, iterations=1, searches=1, 
                    search_type="beam", perturb_type="grad_guided",
                    candidate_type="grad", candidate_grad_delay=0)
            elif args.comb_opt_method == 'genetic':
                target_optimizer = partial(
                    GeneticOptimizer, batch_size=args.batch, candidates=10, 
                    parents=5, generations=10, populations=1)
            elif args.comb_opt_method == 'rand_grad':
                target_optimizer = partial(
                    RandomGradOptimizer, batch_size=args.batch, 
                    candidates=64, iterations=10, searches=1)
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
    
    def __init__(self, modules, shapes, activations, loss_functions, 
                 batch_size, state=[], criterion="loss", regions=10, 
                 requires_grad=False, use_gpu=True):
        self.modules = modules
        self.shapes = [[batch_size] + shape for shape in shapes]
        self.activations = activations
        self.loss_functions = loss_functions
        self.state = {}
        self.criterion = criterion
        self.regions = regions
        self.requires_grad = requires_grad or self.criterion == "loss_grad"
        self.use_gpu = use_gpu

    def generate_target(self, train_step, module_index, 
                        module_target, base_targets=None):
        """
        Subclasses should override this method.
        Args:
            train_step: Int.
            module_index: Int.
            module_target: torch.Tensor; target tensor for module module_index 
                to be used for generating, evaluating, and choosing 
                the targets of module module_index-1.
            base_targets: List; set of target tensors to use for generating 
                new targets (optional, default: None).
        Should return a target torch.Tensor and any relevant evaluation data.
        """
        raise NotImplementedError

    def evaluate_candidates(self, module, module_index, 
                            loss_function, targets, candidates):
        if torch.is_tensor(candidates):
            candidate_batch = candidates
        else:
            candidate_batch = torch.stack(candidates)
        if torch.is_tensor(targets):
            target_batch = targets
        else:
            target_batch = torch.stack(targets)
        candidate_batch = candidate_batch.view(
            candidate_batch.shape[0]*candidate_batch.shape[1], 
            *candidate_batch.shape[2:])
        target_batch = target_batch.view(
            target_batch.shape[0]*target_batch.shape[1], 
            *target_batch.shape[2:])
        candidate_batch = candidate_batch.detach()
        if self.requires_grad:
            candidate_batch.requires_grad_()
        output = module(candidate_batch)
        if module_index > 0:
            output = output.view_as(target_batch)
            if self.criterion != "loss":
                output = self.activations[module_index](output)
            if self.criterion == "accuracy_top5":
                raise RuntimeError("Can only use top 5 accuracy as "
                                   "optimization criteria at output layer.")
        if self.criterion in ["loss", "loss_grad"]:
            losses = loss_function(output, target_batch)
        elif self.criterion == "accuracy":
            losses = accuracy(output, target_batch, average=False)
        elif self.criterion == "accuracy_top5":
            losses = accuracy_topk(output, target_batch, average=False)
        losses = losses.view(len(candidates), int(np.prod(targets.shape[1:])))
        losses = losses.mean(dim=1)  # mean everything but candidate batch dim
        if self.requires_grad:
            losses.sum(dim=0).backward()
            loss_grad = candidate_batch.grad.view_as(candidates)
        if self.criterion == "loss_grad":
            loss_grad = loss_grad.view(loss_grad.shape[0], 
                                       int(np.prod(loss_grad.shape[1:])))
            truncated_length = self.regions*(loss_grad.shape[1]//self.regions)
            loss_grad = loss_grad.resize_(loss_grad.shape[0], truncated_length) 
            loss_grad = loss_grad.view(loss_grad.shape[0], self.regions, 
                                       loss_grad.shape[1]//self.regions)
            loss_grad = loss_grad.mean(dim=2)
        if self.requires_grad:
            return losses, loss_grad
        else:
            return losses

    def step(self, train_step, module_index, module_target, base_targets=None):
        return self.generate_target(train_step, module_index, 
                                    module_target, base_targets)


def generate_neighborhood(base_tensor, size, radius, sampling_weights=None, 
                          perturb_base_tensor=None, use_gpu=True):
    """
    Args:
        base_tensor: torch.LongTensor; base tensor whose 
            neighborhood is generated.
        size: Int; size of the neighborhood, that is, the number of 
            tensors to generate. 
        radius: Int; (expected) number of values to perturb in 
            generating each neighbor. 
        sampling_weights: torch.FloatTensor; weights for randomly sampling 
            indices to perturb (optional, default: None).
        perturb_base_tensor: torch.FloatTensor; base tensor for generating 
            perturbations (optional, default: None).
        use_gpu: Boolean; indicates whether to generate the neighborhoods 
            on the a GPU device or not (optional, default: True).  
    Returns:
        A total of 'size' randomly-generated neighbors of base_tensor. 
    """
    device = torch.device("cuda:0") if use_gpu else torch.device("cpu")
    batch_base = torch.stack([base_tensor]*size)
    sampling_prob = radius / base_tensor.numel()
    if perturb_base_tensor is None:
        one_tensor = torch.ones(batch_base.shape, device=device)
    else:
        one_tensor = perturb_base_tensor
    indices = torch.bernoulli(one_tensor*sampling_prob)
    if sampling_weights is not None:
        batch_weights = torch.stack([sampling_weights]*size)
        sampling_mask = indices.float() * batch_weights
        indices = torch.bernoulli(sampling_mask)  
    neighbourhood = batch_base * (indices*-2 + 1)
    return neighbourhood


def get_random_tensor(shape, make01=False, use_gpu=True):
    device = torch.device("cuda:0") if use_gpu else torch.device("cpu")
    candidate = torch.randint(0, 2, shape, device=device)
    if not make01:
        candidate.mul_(2)
        candidate.add_(-1)
    return candidate


def add_dimensions(tensor, k):
    if tensor.dim() != 2:
        raise NotImplementedError("Not implemented for tensors of dimension != 2!")
    if k == 1:
        expanded_tensor = tensor[:, :, None]
    elif k == 2:
        expanded_tensor = tensor[:, :, None, None]
    elif k == 3:
        expanded_tensor = tensor[:, :, None, None, None]
    elif k == 4:
        expanded_tensor = tensor[:, :, None, None, None, None]
    else:
        raise NotImplementedError("Can't add more than 4 additional dimensions!")
    return expanded_tensor


def multi_split(tensors, sizes):
    return tuple(torch.split(tensor, sizes) for tensor in tensors)


def splice(tensor, region_indices):
    regions = region_indices.shape[1]
    tensor_trunc = tensor.clone().resize_(
        tensor.shape[0], 
        regions*(tensor.shape[1]//regions), 
        *tensor.shape[2:])
    missing_indices = tensor.shape[1] % regions
    tensor_trunc = tensor_trunc.view(
        tensor_trunc.shape[0], regions, 
        tensor_trunc.shape[1]//regions,
        *tensor_trunc.shape[2:])
    region_indices_expand = add_dimensions(
        region_indices, tensor_trunc.dim()-2)
    region_indices_expand = region_indices_expand.expand(
        *region_indices.shape[:2], *tensor_trunc.shape[2:])
    tensor_regions = torch.gather(
        tensor_trunc, 0, region_indices_expand)
    spliced_trunc = tensor_regions.view(
        tensor_regions.shape[0], 
        tensor_regions.shape[1]*tensor_regions.shape[2], 
        *tensor_regions.shape[3:])
    spliced = torch.cat(
        [spliced_trunc, tensor[region_indices[:,-1], -missing_indices:]], dim=1)
    return spliced


def normalized_diff_old(x, y):
    sign_match = torch.eq(x.sign(), y.sign()).float()
    return sign_match * (torch.abs(x + y) / (2*torch.max(torch.abs(x), torch.abs(y))))    


def normalized_diff(x, y):
    sign_match = torch.eq(x.sign(), y.sign()).float()
    shifted_norm = torch.abs(x) + (3/5)
    return sign_match * torch.clamp(shifted_norm, 0, 1)


class SearchOptimizer(TargetPropOptimizer):

    def __init__(self, *args, perturb_scheduler=lambda x, y, z: 1000, candidates=64, 
                 iterations=10, searches=1, search_type="beam", 
                 perturb_type="random", candidate_type="random", 
                 candidate_grad_delay=0, **kwargs):
        super().__init__(*args, **kwargs)
        self.perturb_scheduler = perturb_scheduler
        self.candidates = candidates
        self.iterations = iterations
        self.searches = searches
        self.search_type = search_type
        self.perturb_type = perturb_type
        self.candidate_type = candidate_type
        self.candidate_grad_delay = candidate_grad_delay
        if self.perturb_type == "grad_guided" or self.candidate_type == "grad":
            self.requires_grad = True
        self.state = {"chosen_groups": []}
        # To save a little time, create perturbation tensors beforehand:
        self.perturb_base_tensors = [
            torch.ones([self.candidates] + shape, device=torch.device("cuda:0"))
            for shape in self.shapes]

    def generate_candidates(self, parent_candidates, parameters, 
                            base_perturb_tensors=None, sampling_weights=None):
        candidates = []
        data = enumerate(zip(parent_candidates, parameters, base_perturb_tensors))
        for i, (candidate, (count, perturb_size), perturb_tensor) in data:
            if count == 0:
                raise RuntimeError("Candidate generation count should be nonzero.")
            else:
                sample_tensor = (sampling_weights[i] 
                    if sampling_weights is not None else None)
                candidates.append(candidate.unsqueeze(0))
                candidates.append(generate_neighborhood(
                    candidate, count, perturb_size, 
                    perturb_base_tensor=perturb_tensor, 
                    sampling_weights=sample_tensor))
        return torch.cat(candidates)

    def find_candidates(self, module_index, module_target, 
                        train_step, base_candidates=None):
        loss_function = self.loss_functions[module_index]
        module = self.modules[module_index]
        if module_index > 0:
            module_target = module_target.float()
            loss_function = partial(soft_hinge_loss, reduce_=False)
        target_batch = torch.stack([module_target]*(self.candidates 
            + (len(base_candidates) if base_candidates is not None else 1)))

        search_parameters = [(round(self.candidates*weight), (self.perturb_scheduler
                             if perturb_scheduler is None else perturb_scheduler))
                             for _, weight, perturb_scheduler in base_candidates]
        parents = [get_random_tensor(self.shapes[module_index], use_gpu=self.use_gpu)
                   if candidate is None else candidate 
                   for candidate, _, _ in base_candidates]
        beam_size = len(parents) if self.search_type == "beam" else 1
        if len(parents) > 1:
            perturb_base_tensors = [torch.ones([count] + self.shapes[module_index], 
                                               device=torch.device("cuda:0"))
                                    for count, _ in search_parameters]
        else:
            perturb_base_tensors = [self.perturb_base_tensors[module_index]]

        # Search to find good candidate
        sampling_weights = None
        for i in range(self.iterations):
            iteration_parameters = [
                (count, perturb_scheduler(train_step, module_index, i))
                for count, perturb_scheduler in search_parameters]
            if self.candidate_grad_delay == i and self.candidate_type == "grad":
                if i == 0:
                    # Assumes first parent is activation
                    loss_grad = [self.evaluate_candidates(
                        module, module_index, loss_function, 
                        module_target.unsqueeze(0), parents[0].unsqueeze(0))[1]]
                else:
                    loss_grad = group_data[2]
                candidate_batch = -torch.sign(torch.cat(loss_grad, dim=0))
                parents = ([torch.sign(torch.mean(candidate_batch, 0))] 
                           + (parents[1:] if i == 0 else []))
                best_candidates = [(parent, None) for parent in parents]
                continue
            else:
                candidate_batch = self.generate_candidates(
                    parents, iteration_parameters, 
                    perturb_base_tensors, sampling_weights)
            loss_data = self.evaluate_candidates(module, module_index, loss_function, 
                                                 target_batch, candidate_batch)
            group_data = multi_split(
                (candidate_batch,) + (loss_data 
                    if isinstance(loss_data, tuple) else (loss_data,)), 
                [count+1 for count, _ in iteration_parameters])
            candidate_batches, loss_data = group_data[0], list(zip(*group_data[1:]))
            best_candidates = self.choose_candidates(
                loss_data, candidate_batches, beam_size)
            if self.perturb_type == "grad_guided":
                sampling_weights = [normalized_diff(loss_grad, candidate) 
                                    for candidate, _, loss_grad in best_candidates]
                best_candidates = [(candidate, loss) 
                                   for candidate, loss, _ in best_candidates]
            parents = [candidate for candidate, loss in best_candidates]
        for candidate, loss in best_candidates:
            self.state["candidates"].append((candidate.long(), loss))

    def generate_target(self, train_step, module_index, 
                        module_target, base_targets=None):
        self.state["candidates"] = []
        for i in range(self.searches):
            self.find_candidates(module_index, module_target, 
                                 train_step, base_targets)
        if self.criterion == "loss":
            index, loss = min(
                enumerate([loss for candidate, loss in self.state["candidates"]]),
                key=lambda element: element[1].item() if element[1] is not None else -1)
        else:
            index, loss = 0, None 
        if self.search_type == "parallel":
            self.state["chosen_groups"].append(index)
        return self.state["candidates"][index][0], loss

    def choose_candidates(self, loss_data, candidate_batches, beam_size):
        targets = []
        for loss_tuple, candidates in zip(loss_data, candidate_batches):
            # First element of loss_data should be loss tensor
            losses = loss_tuple[0]
            top_losses, candidate_indices = torch.topk(
                losses, beam_size, dim=0, largest=False, sorted=True)
            if self.criterion == "loss_grad":
                # loss and loss grad
                loss_grad = loss_tuple[1]
                if beam_size > 1:
                    raise NotImplementedError("Loss grad eval metric can currently "
                                              "only be used with beam size == 1.")
                grad_norm = torch.abs(loss_grad)
                loss_factors = torch.stack([losses]*self.regions, dim=1) 
                loss_factors = loss_factors / torch.min(loss_factors)
                grad_norm = grad_norm * loss_factors  # accounts for loss differences
                top_grad_norm, top_region_indices = torch.min(
                    grad_norm, dim=0, keepdim=True)
                best_candidates = splice(candidates, top_region_indices)
                targets.append((best_candidates.squeeze(0), None))
            else:
                best_candidates = candidates[candidate_indices]
                if self.perturb_type == "grad_guided":
                    top_loss_grads = loss_tuple[1][candidate_indices]
                    targets.append((best_candidates.squeeze(0), 
                                    top_losses, top_loss_grads.squeeze(0)))
                else:
                    targets.append((best_candidates.squeeze(0), top_losses))
        return targets


class RandomGradOptimizer(TargetPropOptimizer):

    def __init__(self, *args, candidates=64, iterations=10, searches=1, **kwargs):
        super().__init__(*args, **kwargs)
        self.candidates = candidates
        self.iterations = iterations
        self.searches = searches  
        self.requires_grad = True
        self.state = {"chosen_groups": []}

    def find_candidates(self, module_index, module_target, 
                        train_step, base_candidates=None):
        loss_function = self.loss_functions[module_index]
        module = self.modules[module_index]
        if module_index > 0:
            module_target = module_target.float()
            loss_function = partial(soft_hinge_loss, reduce_=False)
        target_batch = torch.stack([module_target]*self.candidates)

        parents = [get_random_tensor(self.shapes[module_index], use_gpu=self.use_gpu)
                   for i in range(self.candidates)]
        candidate_batch = torch.stack(parents, dim=0)
        for i in range(self.iterations):
            loss_grad = self.evaluate_candidates(
                module, module_index, loss_function, target_batch, candidate_batch)[1]
            if i == 0:
                no_threshold_batch = -loss_grad
            else:
                learning_rate = (self.iterations - i) / self.iterations
                no_threshold_batch = no_threshold_batch - (learning_rate*loss_grad)
            candidate_batch = torch.sign(no_threshold_batch)
        candidate = torch.sign(torch.mean(candidate_batch, 0))
        self.state["candidates"].append(candidate.long())

    def generate_target(self, train_step, module_index, 
                        module_target, base_targets=None):
        self.state["candidates"] = []
        for i in range(self.searches):
            self.find_candidates(module_index, module_target, 
                                 train_step, base_targets)
        return self.state["candidates"][0], None


def print_param_info(module, layer):
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
            shapes=model.input_sizes[::-1],
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
            with torch.no_grad():
                evaluate(model, eval_dataset_loader, loss_function, 
                         eval_metrics, logger, 0, log=True, use_gpu=use_gpu)
        last_time = time()
        model.train()
        for i, batch in enumerate(train_dataset_loader):
            inputs, labels = batch[0], batch[1]
            if inputs.shape[0] != batch_size:
                continue
            if use_gpu:
                inputs, labels = inputs.cuda(), labels.cuda()
            if isinstance(model, ToyNet):
                try:
                    inputs = inputs.view(batch_size, 784)
                except RuntimeError:
                    try:
                        inputs = inputs.view(batch_size, 15)
                    except RuntimeError:
                        inputs = inputs.view(batch_size, 3072)
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
                        loss = loss_function(outputs, targets.detach()).mean()
                        output_loss = loss
                    else:
                        loss = soft_hinge_loss(outputs, targets.detach().float()).mean()
                    loss.backward()
                    optimizer.step()
                    if j == 1 and i % 5 == 1 and False:
                        # Check loss change after SGD update step
                        updated_loss = loss_function(model(inputs), labels).mean()
                        loss_delta = updated_loss - output_loss
                        logger.scalar_summary(
                            "train/loss_delta", loss_delta.item(), train_step)
                    if j != len(modules)-1:
                        activation = model.all_activations[len(modules)-1-j-1](
                            activations[j+1].detach())
                        if args.model == "convnet4":
                            perturb_sizes = [7000, 15000, 30000]
                        elif args.model == "toynet":
                            perturb_sizes = [1000]
                        perturb_scheduler = lambda x, y, z: perturb_sizes[-y-1]
                        targets, target_loss = target_optimizer.step(
                            train_step, j, targets, 
                            base_targets=[[None, 1/1, perturb_scheduler]])
                    if print_param_info and i % 100 == 1:
                        print_param_metrics(module, len(modules)-1-j)
            if (train_step in [0, 10, 100, 1000, 10000, 50000] 
                    and args.model == "toynet" and args.collect_params):
                target_loss = loss_function(modules[0][0](targets.float()), labels)
                store_step_data(model, labels[10], target_loss[10].item(), train_step, 
                                "model_data\\" + args.model + "_" + args.dataset
                                + "bs" + str(args.batch) + "_" + args.nonlin 
                                + "_" + ("comb" if args.comb_opt else "grad"))
            if not train_per_layer:
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
        with torch.no_grad():
            evaluate(model, eval_dataset_loader, loss_function, 
                     eval_metrics, logger, train_step, log=True, use_gpu=use_gpu)
        if args.store_checkpoints:
            store_checkpoint(model, optimizer, args, training_metrics, 
                             eval_metrics, epoch, os.path.join(os.curdir, 'new_logs'))


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
        if use_gpu:
            inputs, labels = inputs.cuda(), labels.cuda()
        if isinstance(model, ToyNet):
            try:
                inputs = inputs.view(batch_size, 784)
            except RuntimeError:
                try:
                    inputs = inputs.view(batch_size, 15)
                except RuntimeError:
                    inputs = inputs.view(batch_size, 3072)
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


def correct_format(row_string):
    # Remove delimiters and correct spacing
    corrected_string = row_string.replace("[", "").replace("]", "").strip()
    corrected_string = re.sub(r",(?: |  )", " ", corrected_string)
    # Remove scientific notation
    values = re.split(r" +", corrected_string)
    corrected_string = " ".join([str(float(value)) for value in values])
    corrected_string = re.sub(r" (?!-)", "  ", corrected_string)
    return corrected_string


def store_step_data(model, label, target_loss, train_step, file_prefix, layer=2):
    torch.set_printoptions(threshold=1000000, linewidth=1000000)
    parameters = list(model.parameters())
    weight_matrix = parameters[2*(layer-1)].transpose(0, 1)
    bias_vector = parameters[2*(layer-1)+1]
    weight_rows = re.findall(r"\[.*?\]", str(weight_matrix))
    bias_rows = re.findall(r"\[.*?\]", str(bias_vector))
    weight_file_name = file_prefix + "_weight_step" + str(train_step) + ".txt"
    with open(weight_file_name, "w+") as weight_file:
        for row in weight_rows:
            print(correct_format(row), file=weight_file)
    bias_file_name = file_prefix + "_bias_step" + str(train_step) + ".txt"
    with open(bias_file_name, "w+") as bias_file:
        for row in bias_rows: 
            print(correct_format(row), file=bias_file)
    if label.numel() == 1:
        label_vector = torch.zeros(10)
        label_vector[label] = 1
        label = label_vector
    label_rows = re.findall(r"\[.*?\]", str(label))
    label_file_name = file_prefix + "_target_step" + str(train_step) + ".txt"
    with open(label_file_name, "w+") as label_file:
        for row in label_rows:
            print(correct_format(row), file=label_file)
    loss_file_name = file_prefix + "_loss_step" + str(train_step) + ".txt"
    with open(loss_file_name, "w+") as loss_file:
        print(target_loss, file=loss_file)
    torch.set_printoptions(profile="default")


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
