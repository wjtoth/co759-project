# python
import os
import re
import argparse
from functools import partial
from itertools import chain
from time import time

# pytorch
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
import torch.multiprocessing as multiprocessing
from torch.multiprocessing import Process

# Friesen and Domingos
import activations
import targetprop as tp
from datasets import create_datasets
from util.tensorboardlogger import TensorBoardLogger

# ours
import adversarial
from graph_nn import get_graphs
from dropbox_tools import upload
from search_optimize import (soft_hinge_loss, accuracy, accuracy_topk, 
                             SearchOptimizer, RandomGradOptimizer)
from search_models import ToyNet, ConvNet4, ConvNet8, Step


def get_args():
    # argument definitions
    parser = argparse.ArgumentParser()

    # model arguments
    parser.add_argument("--model", type=str, default="convnet4",
                        choices=("convnet4", "convnet8", "toynet"))
    parser.add_argument("--nonlin", type=str, default="relu",
                        choices=("relu", "step01", "step11", "staircase"))

    # training/optimization arguments
    parser.add_argument("--no-train", action="store_true", default=False)
    parser.add_argument("--batch", type=int, default=64,
                        help="batch size to use for training")
    parser.add_argument("--epochs", type=int, default=10,
                        help="number of epochs to train for")
    parser.add_argument("--test-batch", type=int, default=0,
                        help="batch size to use for validation and testing")
    parser.add_argument("--loss", type=str, default="cross_entropy",
                        choices=("cross_entropy", "hinge"))
    parser.add_argument("--wtdecay", type=float, default=0)
    parser.add_argument("--lr-decay-factor", type=float, default=1.0,
                        help="factor by which to multiply the learning rate " 
                             "at each value in <lr-decay-epochs>")
    parser.add_argument("--lr-decay-epochs", type=int, nargs="+", default=None,
                        help="list of epochs at which to multiply " 
                             "the learning rate by <lr-decay>")
    parser.add_argument("--collect-params", action="store_true", default=False,
                        help="if specified, collects model data "
                             "at various training steps for AMPL optimization; "
                             "currently only available for ToyNet model.")

    # target propagation arguments
    parser.add_argument("--grad-tp-rule", type=str, default="SoftHinge",
                        choices=("WtHinge", "TruncWtHinge", "SoftHinge", 
                                 "STE", "SSTE", "SSTEAndTruncWtHinge"))
    parser.add_argument("--softhinge-factor", type=float, default=1.0)
    parser.add_argument("--comb-opt", action="store_true", default=False,
                        help="if specified, combinatorial optimization methods " 
                             "are used for target setting")
    parser.add_argument("--comb-opt-method", type=str, default="local_search",
                        choices=("local_search", "genetic", "rand_grad"))
    parser.add_argument("--target-momentum", action="store_true", default=False,
                        help="if specified, target momentum is used. "
                             "Note: only supported with gradient-based targetprop")
    parser.add_argument("--target-momentum-factor", type=float, default=0.0,
                        help="factor by which to multiply the momentum tensor "
                             "during target setting")
    parser.add_argument("--no-loss-grad-weight", action="store_true", default=False)

    # combinatorial search arguments
    parser.add_argument("--criterion", type=str, default="loss", 
                        choices=("loss", "output_loss", "loss_grad", 
                                 "accuracy", "accuracy_top5"))
    parser.add_argument("--no-criterion-grad-weight", action="store_true", default=False)
    parser.add_argument("--candidates", type=int, default=64)
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--searches", type=int, default=1)
    parser.add_argument("--search-type", type=str, default="beam", 
                        choices=("beam", "parallel"))
    parser.add_argument("--perturb-type", type=str, default="random",
                        choices=("random", "grad_guided"))
    parser.add_argument("--perturb-sizes", type=int, default=None, nargs="+")
    parser.add_argument("--candidate-type", type=str, default="random",
                        choices=("random", "grad", "grad_sampled"))
    parser.add_argument("--candidate-grad-delay", type=int, default=1)

    # data arguments
    parser.add_argument("--dataset", type=str, default="cifar10",
                        choices=("mnist", "cifar10", "cifar100", 
                                 "svhn", "imagenet", "graphs"))
    parser.add_argument("--data-root", type=str, default="",
                        help="root directory for imagenet dataset "
                             "(with separate train, val, test folders)")     
    parser.add_argument("--no-aug", action="store_true", default=False,
                        help="if specified, do not use data augmentation " 
                             "(default=True for MNIST, False for CIFAR10)")
    parser.add_argument("--download", action="store_true",
                        help="allow downloading of the dataset " 
                             "(not including imagenet) if not found")
    parser.add_argument("--dbg-ds-size", type=int, default=0,
                        help="debug: artificially limit the size of the training data")
    parser.add_argument("--nworkers", type=int, default=2,
                        help="number of workers to use for loading data from disk")
    parser.add_argument("--no-val", action="store_true", default=False,
                        help="if specified, do not create a validation set from "
                             "the training data and use it to choose the best model")

    # adversarial evaluation arguments
    parser.add_argument("--adv-eval", action="store_true", default=False, 
                        help="if specified, evaluates the network on " 
                             "adversarial examples generated using adv-attack")
    parser.add_argument("--adv-attack", type=str, default="fgsm", 
                        choices=tuple(adversarial.ATTACKS.keys()) + ("all",))
    parser.add_argument("--adv-epsilon", type=float, default=0.25)

    # computation arguments
    parser.add_argument("--runs", type=int, default=1)
    parser.add_argument("--cuda", action="store_true", default=False,
                        help="if specified, use CPU only")
    parser.add_argument("--device", type=int, default=0, help="CUDA device ID to use")
    parser.add_argument("--multi-gpu", action="store_true", default=False,
                        help="if specified, use all available GPUs")
    parser.add_argument("--seed", type=int, default=468412397,
                        help="random seed")

    # logging arguments
    parser.add_argument("--tb-logging", action="store_true", default=False,
                        help="if specified, enable logging Tensorboard summaries")
    parser.add_argument("--store-checkpoints", action="store_true", default=False, 
                        help="if specified, enables storage of the current model " 
                             "and training parameters at each epoch")
    parser.add_argument("--logs-to-dbx", action="store_true", default=False,
                        help="if specified, uploads all local log files to dropbox")
    parser.add_argument("--dbx-token", type=str)

    args = parser.parse_args()
    args.cuda = args.cuda and torch.cuda.is_available()
    args.lr_decay_epochs = [] if args.lr_decay_epochs is None else args.lr_decay_epochs
    if not args.no_aug:
        args.no_aug = True if args.dataset == "mnist" else False
    args.grad_tp_rule = tp.TPRule[args.grad_tp_rule]
    args.perturb_sizes = (args.perturb_sizes if isinstance(args.perturb_sizes, list) 
                          or args.perturb_sizes is None else [args.perturb_sizes])
    
    return args


def main(args):
    if args.dataset == "graphs":
        train_loader, val_loader = get_graphs(
            6, batch_size=args.batch, num_workers=args.nworkers)
        num_classes = 2
    else:
        train_loader, val_loader, test_loader, num_classes = \
            create_datasets(args.dataset, args.batch, args.test_batch, not args.no_aug, 
                            args.no_val, args.data_root, args.cuda, args.seed, 
                            args.nworkers, args.dbg_ds_size, args.download)
    if args.adv_eval:
        adversarial_eval_dataset = get_adversarial_dataset(args)

    args.nonlin = args.nonlin.lower()
    if args.nonlin == "relu":
        nonlin = nn.ReLU
    elif args.nonlin == "step01":
        nonlin = partial(Step, make01=True, targetprop_rule=args.grad_tp_rule, 
                         use_momentum=args.target_momentum, 
                         momentum_factor=args.target_momentum_factor,
                         scale_by_grad_out=not args.no_loss_grad_weight,
                         tanh_factor=args.softhinge_factor)
    elif args.nonlin == "step11":
        nonlin = partial(Step, make01=False, targetprop_rule=args.grad_tp_rule,
                         use_momentum=args.target_momentum, 
                         momentum_factor=args.target_momentum_factor, 
                         scale_by_grad_out=not args.no_loss_grad_weight,
                         tanh_factor=args.softhinge_factor)
    elif args.nonlin == "staircase":
        nonlin = partial(activations.Staircase, targetprop_rule=args.grad_tp_rule,
                         nsteps=5, margin=1, trunc_thresh=2,
                         scale_by_grad_out=not args.no_loss_grad_weight)

    if args.dataset == "mnist":
        input_shape = (1, 28, 28)
        if args.model == "toynet":
            input_shape = (784,)
    elif args.dataset.startswith("cifar"):
        input_shape = (3, 32, 32)
        if args.model == "toynet":
            input_shape = (3072,)
    elif args.dataset == "svhn":
        input_shape = (3, 40, 40)
    elif args.dataset == "imagenet":
        input_shape = (3, 224, 224)
    elif args.dataset == "graphs":
        input_shape = (15,)
    else:
        raise NotImplementedError("no other datasets currently supported")

    print("Creating network...")
    if args.model == "convnet4":
        network = ConvNet4(nonlin=nonlin, input_shape=input_shape, 
                           separate_activations=args.comb_opt, 
                           multi_gpu_modules=args.multi_gpu)
    elif args.model == "convnet8":
        network = ConvNet8(nonlin=nonlin, input_shape=input_shape, 
                           separate_activations=args.comb_opt, 
                           multi_gpu_modules=args.multi_gpu)
    elif args.model == "toynet":
        if args.dataset not in ["cifar10", "mnist", "graphs"]:
            raise NotImplementedError(
                "Toy network can only be trained on CIFAR10, "
                "MNIST, or graph connectivity task.")
        network = ToyNet(nonlin=nonlin, input_shape=input_shape,
                         separate_activations=args.comb_opt, 
                         num_classes=num_classes, 
                         multi_gpu_modules=args.multi_gpu)
    network.needs_backward_twice = False
    if args.nonlin.startswith("step") or args.nonlin == "staircase":
        network.targetprop_rule = args.grad_tp_rule
        network.needs_backward_twice = tp.needs_backward_twice(args.grad_tp_rule)
    if args.cuda:
        print("Moving to GPU...")
        with torch.cuda.device(args.device):
            network = network.cuda()
    if args.multi_gpu and not args.comb_opt:
        network = nn.DataParallel(network)

    print("Setting up logging...")
    log_dir = os.path.join(os.path.join(os.curdir, "logs"), str(round(time())))
    tb_logger = TensorBoardLogger(log_dir) if args.tb_logging else None

    if args.no_train:
        print("Loading from last checkpoint...")
        checkpoint_state = load_checkpoint(os.path.join(os.curdir, "logs"))
        model_state = checkpoint_state["model_state"]
        network.load_state_dict(model_state)
    else:
        print("Creating loss function...")
        if args.loss == "cross_entropy":
            criterion = torch.nn.CrossEntropyLoss(
                size_average=not args.comb_opt, reduce=not args.comb_opt)
        elif args.loss == "hinge":
            criterion = partial(multiclass_hinge_loss, reduce_=not args.comb_opt)
        print("Creating parameter optimizer...")
        optimizer = partial(optim.Adam, lr=0.00025, weight_decay=args.wtdecay)
        if args.comb_opt:
            print("Creating target optimizer...")
            if args.nonlin != "step11":
                raise NotImplementedError(
                    "Discrete opt methods currently only support nonlin = step11.")
            if args.comb_opt_method == "local_search":
                target_optimizer = partial( 
                    SearchOptimizer, batch_size=args.batch, 
                    criterion=args.criterion, regions=10, 
                    perturb_scheduler=partial(perturb_scheduler, [1000]), 
                    candidates=args.candidates, iterations=args.iterations, 
                    searches=args.searches, search_type=args.search_type, 
                    perturb_type=args.perturb_type, candidate_type=args.candidate_type, 
                    candidate_grad_delay=args.candidate_grad_delay)
            elif args.comb_opt_method == "genetic":
                raise NotImplementedError(
                    "Pure genetic algorithm not currently implemented.")
            elif args.comb_opt_method == "rand_grad":
                target_optimizer = partial(
                    RandomGradOptimizer, batch_size=args.batch, 
                    candidates=args.candidates, iterations=args.iterations, 
                    searches=args.searches)
            else:
                raise NotImplementedError
        else:
            target_optimizer = None

        with torch.cuda.device(args.device):
            try:
                train(network, train_loader, val_loader, criterion, optimizer, 
                      args.epochs, num_classes, target_optimizer=target_optimizer, 
                      log_dir=log_dir, logger=tb_logger, use_gpu=args.cuda, args=args)
            except KeyboardInterrupt:
                print("\nTraining interrupted!\n")
            else:
                print("\nFinished training.\n")

    if args.adv_eval:
        print("Evaluating on adversarial examples...")
        if args.adv_attack == "all":
            for attack in adversarial.ATTACKS:
                with torch.cuda.device(args.device):
                    failure_rate = evaluate_adversarially(
                        network, adversarial_eval_dataset, "untargeted_misclassify", 
                        attack, args.test_batch, num_classes, 
                        args.adv_epsilon, args.cuda)
                print("Failure rate: %0.2f%%" % (100*failure_rate))
        else:
            with torch.cuda.device(args.device):
                failure_rate = evaluate_adversarially(
                    network, adversarial_eval_dataset, "untargeted_misclassify", 
                    args.adv_attack, args.test_batch, num_classes, 
                    args.adv_epsilon, args.cuda)
            print("Failure rate: %0.2f%%" % (100*failure_rate))

    if args.logs_to_dbx:
        upload(log_dir, token=args.dbx_token)


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


def print_param_info(module, layer):
    for k, param in enumerate(module.parameters()):
        if k == 0:
            print("\nlayer {0} - weight matrices mean and variance: "
                  "{1:.8f}, {2:.8f}\n".format(
                  layer, torch.mean(param.data), 
                  torch.std(param.data)))
            print("layer {0} - gradient mean and variance: "
                  "{1:.8f}, {2:.8f}\n".format(
                  layer, torch.mean(param.grad.data), 
                  torch.std(param.grad.data)))


def train(model, train_dataset_loader, eval_dataset_loader, loss_function, 
          optimizer, epochs, num_classes, target_optimizer=None, 
          log_dir=None, logger=None, use_gpu=True, args=None):
    model.train()
    batch_size = train_dataset_loader.batch_size
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
        modules = None
        optimizer = optimizer(model.parameters())
        lr_schedulers = [optim.lr_scheduler.MultiStepLR(
            optimizer, args.lr_decay_epochs, gamma=args.lr_decay_factor)]
        if args.target_momentum:
            activations = list(chain.from_iterable(
                [[child for i, child in enumerate(module) if i == len(module)-1] 
                 for module in model.all_modules[:-1]]))
            for i, activation in enumerate(activations):
                activation.initialize_momentum_state(
                    [batch_size] + model.input_sizes[i+1], num_classes)
    print("Optimizers created.")
    extra_args = target_optimizer.__dict__.copy() if args.comb_opt else None
    if extra_args is not None and "perturb_base_tensors" in extra_args:
        del extra_args["perturb_base_tensors"]
    store_run_info(args, log_dir, extra_args=extra_args)
    training_metrics = Metrics({"dataset_batches": len(train_dataset_loader),
                                "loss": [], "accuracy": [], 
                                "accuracy_top5": [], "steps/sec": []})
    eval_metrics = Metrics({"dataset_batches": len(eval_dataset_loader), 
                            "loss": [], "accuracy": [], "accuracy_top5": []})
    print("Beginning training...")
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
            step = epoch*len(train_dataset_loader) + i
            if i % 10 == 1:
                last_time = eval_step(model, inputs, labels, loss_function, 
                                      training_metrics, logger, epoch, 
                                      i, step, last_time)
            train_step(model, modules, inputs, labels, loss_function, optimizer, 
                       target_optimizer, args, step, train_per_layer, log_dir)
        with torch.no_grad():
            evaluate(model, eval_dataset_loader, loss_function, 
                     eval_metrics, logger, step, log=True, use_gpu=use_gpu)
        if args.store_checkpoints:
            store_checkpoint(model, optimizers if train_per_layer else optimizer, 
                             args, training_metrics, eval_metrics, epoch, log_dir)


def perturb_scheduler(train_step, module_index, iteration, perturb_sizes=[1000]):
    return perturb_sizes[-module_index-1]


def train_step(model, modules, inputs, targets, loss_function, optimizer, 
               target_optimizer, args, step, train_per_layer, log_dir):
    output_targets = targets
    if train_per_layer:
        # Obtain activations / hidden states
        preactivations = []
        postactivations = []
        for j, (module, _) in enumerate(modules[::-1]):
            if j == 0:
                outputs = module(inputs)
            else:
                outputs = module(activation)
            preactivations.append(outputs)
            if j != len(modules)-1:
                activation = model.all_activations[j](outputs).detach()
                activation.requires_grad_()
                postactivations.append(activation)
        preactivations.reverse()
        postactivations.reverse()
        # Then target prop in reverse mode
        loss_grad = 1
        for j, (module, optimizer) in enumerate(modules):
            optimizer.zero_grad()
            outputs = preactivations[j]
            if j == 0:
                loss = loss_function(outputs, targets.detach()).mean()
            else:
                loss = soft_hinge_loss(
                    outputs, targets.detach().float(), reduce_=False)
                if not args.no_loss_grad_weight:
                    loss *= torch.abs(loss_grad)
                loss = loss.sum()
            if j != len(modules)-1:
                activation = model.all_activations[len(modules)-1-j-1](
                    preactivations[j+1].detach())
                if args.model == "convnet4":
                    perturb_sizes = args.perturb_sizes or [7000, 15000, 30000]
                elif args.model == "toynet":
                    perturb_sizes = args.perturb_sizes or [1000]
                perturb_schedule = partial(
                    perturb_scheduler, perturb_sizes=perturb_sizes)
                targets, target_loss = target_optimizer.step(
                    step, j, targets, 
                    base_targets=[[None, 1/1, perturb_schedule]], 
                    criterion_weight=(None if j == 0 or args.no_criterion_grad_weight 
                                      else torch.abs(loss_grad)))
            optimizer.zero_grad()
            loss.backward()
            if j != len(modules)-1:
                loss_grad = postactivations[j].grad
            optimizer.step()
            
    if (step in [0, 10, 100, 1000, 10000, 50000] 
            and args.model == "toynet" and args.collect_params):
        target_loss = loss_function(modules[0][0](targets.float()), output_targets)
        store_step_data(model, output_targets[10], target_loss[10].item(), 
                        step, os.path.join(log_dir, "run_data"))
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


def train_step_grad(model, modules, inputs, targets, loss_function, optimizer, 
                    target_optimizer, args, step, train_per_layer, log_dir):
    if train_per_layer:
        # Obtain activations / hidden states
        preactivations = []
        postactivations = []
        for j, (module, _) in enumerate(modules[::-1]):
            if j == 0:
                outputs = module(inputs)
            else:
                outputs = module(activation)
            preactivations.append(outputs)
            if j != len(modules)-1:
                activation = model.all_activations[j](outputs).detach()
                activation.requires_grad_()
                postactivations.append(activation)
        preactivations.reverse()
        postactivations.reverse()
        # Then target prop in reverse mode
        loss_grad = 1
        for j, (module, optimizer) in enumerate(modules):
            optimizer.zero_grad()
            outputs = preactivations[j]
            if j == 0:
                loss = loss_function(outputs, targets.detach()).mean()
            else:
                #loss = torch.tanh(outputs) * loss_grad
                loss = soft_hinge_loss(
                    outputs, targets.detach().float(), reduce_=False)
                loss *= torch.abs(loss_grad)
                loss = loss.sum()
            optimizer.zero_grad()
            loss.backward()
            if j != len(modules)-1:
                loss_grad = postactivations[j].grad
                targets = -torch.sign(loss_grad)
            optimizer.step()
    else:
        raise RuntimeError("Change the training step function!")


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
    metric_tuple = (("loss", loss), ("accuracy", batch_accuracy), 
                    ("accuracy_top5", batch_accuracy_top5), 
                    ("steps/sec", steps_per_sec))
    training_metrics.append(metric_tuple)
    print("training --- epoch: %d, batch: %d, loss: %.3f, acc: %.3f, "
          "acc_top5: %.3f, steps/sec: %.2f" 
          % (epoch+1, batch+1, loss, batch_accuracy, 
             batch_accuracy_top5, steps_per_sec))
    if logger is not None:
        logger.scalar_summary("train/loss", loss, step)
        logger.scalar_summary("train/accuracy", batch_accuracy, step)
        logger.scalar_summary("train/top5_accuracy", batch_accuracy_top5, step)
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
        print("\nevaluation --- loss: %.3f, acc: %.3f, acc_top5: %.3f \n" 
              % (loss, total_accuracy, total_accuracy_top5))
        if logger is not None:
            logger.scalar_summary("eval/loss", loss, step)
            logger.scalar_summary("eval/accuracy", total_accuracy, step)
            logger.scalar_summary("eval/top5_accuracy", total_accuracy_top5, step)
    eval_metrics.append((("loss", loss), ("accuracy", total_accuracy), 
                         ("accuracy_top5", total_accuracy_top5)))

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


def store_run_info(terminal_args, log_dir, extra_args=None):
    if extra_args is None:
        extra_args = {}
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    hparams = terminal_args.__dict__
    hparams.update(extra_args)
    with open(os.path.join(log_dir, "hparams.txt"), "w+") as info_file:
        print(hparams, file=info_file)


def store_checkpoint(model, optimizers, terminal_args, training_metrics, 
                     eval_metrics, epoch, log_dir):
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    optimizer_states = ([optimizer.state_dict() for optimizer in optimizers]
                        if isinstance(optimizers, list) else optimizers.state_dict())
    checkpoint_state = {
        "args": terminal_args,
        "model_state": model.state_dict(),
        "optimizers": optimizer_states,
        "training_metrics": training_metrics,
        "eval_metrics": eval_metrics,
        "epoch": epoch,
        "save_time": time(),
    }
    file_name = "checkpoint_epoch{}.state".format(epoch)
    file_path = os.path.join(log_dir, file_name)
    torch.save(checkpoint_state, file_path)
    print("\nModel checkpoint saved at: " 
          + "\\".join(file_path.split("\\")[-2:]) + "\n")
    # delete old checkpoint
    if epoch > 9 and epoch % 10 != 0:
        previous = os.path.join(
            log_dir, "checkpoint_epoch{}.state".format(epoch-10))
        if os.path.exists(previous) and os.path.isfile(previous):
            os.remove(previous)


def load_checkpoint(log_dir, epoch=None):
    if epoch is None:
        checkpoint_files = [file_name for file_name in os.listdir(log_dir)
                            if file_name.startswith("checkpoint")]
        checkpoint_files.sort()
    else:
        checkpoint_files = [
            file_name for file_name in os.listdir(log_dir)
            if file_name.startswith("checkpoint_epoch{}".format(epoch))
        ]

    checkpoint_state = torch.load(os.path.join(log_dir, checkpoint_files[-1]))
    return checkpoint_state


def correct_format(row_string, index=None):
    # Remove delimiters and correct spacing
    corrected_string = row_string.replace("[", "").replace("]", "").strip()
    corrected_string = re.sub(r",(?: |  )", " ", corrected_string)
    # Remove scientific notation
    values = re.split(r" +", corrected_string)
    corrected_string = " ".join([str(float(value)) for value in values])
    corrected_string = re.sub(r" (?!-)", "  ", corrected_string)
    # Place in columns
    values = [element for element in corrected_string.split(" ") if element.strip()]
    corrected_string = str(index) + "\t" if index is not None else ""
    for value in values:
        corrected_string += value + "\t"
    return corrected_string


def store_step_data(model, label, target_loss, train_step, log_dir, layer=2):
    if not os.path.exists(log_dir):
        os.mkdir(log_dir)
    torch.set_printoptions(threshold=1000000, linewidth=1000000)
    parameters = list(model.parameters())
    weight_matrix = parameters[2*(layer-1)].transpose(0, 1)
    bias_vector = parameters[2*(layer-1)+1]
    weight_rows = re.findall(r"\[.*?\]", str(weight_matrix))
    bias_rows = re.findall(r"\[.*?\]", str(bias_vector))
    weight_rows = [correct_format(row, index=i+1) for i, row in enumerate(weight_rows)]
    bias_rows = [correct_format(row, index=i+1) for i, row in enumerate(bias_rows)]
    weight_file_name = os.path.join(log_dir, "weight_step" + str(train_step) + ".txt")

    if label.numel() == 1:
        label_vector = torch.zeros(10)
        label_vector[label] = 1
        label = label_vector
    label_rows = re.findall(r"\[.*?\]", str(label))
    label_rows = [correct_format(row, index=i+1) for i, row in enumerate(label_rows)]

    weight_string = "\n".join(weight_rows)
    bias_string = "\n".join(bias_rows)
    label_string = "\n".join(label_rows)
    with open(weight_file_name, "w+") as weight_file:
        print(weight_string, file=weight_file)
    with open(weight_file_name.replace("weight", "bias"), "w+") as bias_file:
        print(bias_string, file=bias_file)
    with open(weight_file_name.replace("weight", "target"), "w+") as label_file:
        print(label_string, file=label_file)

    data_string = "data;\n\nparam M:=100 ;\nparam N:=10 ;\n\n\n"
    data_string += "param T:\n\t" + "\t".join(str(i) for i in range(1, 11)) + " :=\n"
    data_string += label_string + " ;\n\n\n"
    data_string += "param B:\n\t" + "\t".join(str(i) for i in range(1, 11)) + " :=\n"
    data_string += bias_string + " ;\n\n\n"
    data_string += "param W:\n\t\t" + "\t\t".join(str(i) for i in range(1, 11)) + " :=\n"
    data_string += weight_string + "  ;"
    with open(weight_file_name.replace("weight", "data"), "w+") as data_file:
        print(data_string, file=data_file)
    with open(weight_file_name.replace("weight", "loss"), "w+") as loss_file:
        print(target_loss, file=loss_file)
    torch.set_printoptions(profile="default")


if __name__ == "__main__":
    args = get_args()
    main(args)
    # multiprocessing.set_start_method("spawn")
    # args = get_args()
    # for i in range(args.runs):
    #     print("\nStarting training run " + str(i+1) + "...\n")
    #     # Run in separate process to avoid PyTorch multiprocessing errors
    #     process = Process(target=main, args=(args,))
    #     process.start()
    #     process.join()