# python
import os
import re
import argparse
import json
from functools import partial
from itertools import chain
from time import time

# pytorch / numpy
import numpy as np
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
import utils
from graph_nn import get_graphs
from dropbox_tools import upload
from search_optimize import (soft_hinge_loss, accuracy, accuracy_topk, 
                             SearchOptimizer, RandomGradOptimizer)
from search_models import ToyNet, ConvNet4, ConvNet8, Step
from neos import run_neos_job


def get_args():
    # argument definitions
    parser = argparse.ArgumentParser()

    # model arguments
    parser.add_argument("--model", type=str, default="convnet4",
                        choices=("convnet4", "convnet8", "toynet"))
    parser.add_argument("--nonlin", type=str, default="relu",
                        choices=("relu", "step01", "step11", "staircase"))
    parser.add_argument("--hidden-units", type=int, default=100,
                        help="number of hidden units in the toy network")
    parser.add_argument("--no-biases", action="store_true", default=False,
                        help="if specified, the network does not contain any biases"
                             "---only applies to the toy network.")

    # training/optimization arguments
    parser.add_argument("--no-train", action="store_true", default=False)
    parser.add_argument("--test", action="store_true", default=False)
    parser.add_argument("--eval", action="store_true", default=False)
    parser.add_argument("--load-checkpoints", type=str, nargs="+", default=None)
    parser.add_argument("--load-checkpoint-best", action="store_true", default=False)
    parser.add_argument("--batch", type=int, default=64,
                        help="batch size to use for training")
    parser.add_argument("--epochs", type=int, default=10,
                        help="number of epochs to train for")
    parser.add_argument("--test-batch", type=int, default=0,
                        help="batch size to use for validation and testing")
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
    parser.add_argument("--collect-timesteps", type=int, nargs="+", 
                        default=[10, 30, 100, 300, 1000, 3000, 10000, 30000])
    parser.add_argument("--collect-count", type=int, default=1,
                        help="if collect_params, collect data of this many targets")
    parser.add_argument("--ampl-train", action="store_true", default=False)
    parser.add_argument("--ampl-train-steps", type=int, default=625)
    parser.add_argument("--ampl-eval", action="store_true", default=False)
    parser.add_argument("--ampl-eval-dir", type=str, default=None)
    parser.add_argument("--baron-options", type=str, default=None)

    # target propagation arguments
    parser.add_argument("--grad-tp-rule", type=str, default="SoftHinge",
                        choices=("WtHinge", "TruncWtHinge", "SoftHinge", 
                                 "STE", "SSTE", "SSTEAndTruncWtHinge"))
    parser.add_argument("--softhinge-factor", type=float, default=1.0)
    parser.add_argument("--comb-opt", action="store_true", default=False,
                        help="if specified, combinatorial optimization methods " 
                             "are used for target setting")
    parser.add_argument("--comb-opt-method", type=str, default="local_search",
                        choices=("local_search", "rand_grad"))
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
    parser.add_argument("--grad-guide-offset", type=float, default=3/5)
    parser.add_argument("--candidate-type", type=str, default="random",
                        choices=("random", "grad", "grad_sampled"))
    parser.add_argument("--candidate-grad-delay", type=int, default=1)
    parser.add_argument("--splice-conv-targets", action="store_true", default=False)

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
                             "and training parameters at the best-performing "
                             "and final epochs")
    parser.add_argument("--store-many-checkpoints", action="store_true", default=False, 
                        help="if specified, enables storage of the current model " 
                             "and training parameters at every 10th epoch")
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
    if args.baron_options is not None:
        args.baron_options = json.load(args.baron_options)
    
    return args


def main(args):
    # Get datasets
    if args.dataset == "graphs":
        train_loader, val_loader, test_loader = get_graphs(
            6, batch_size=args.batch, num_workers=args.nworkers, 
            val_set=not args.no_val)
        num_classes = 2
    else:
        train_loader, val_loader, test_loader, num_classes = \
            create_datasets(args.dataset, args.batch, args.test_batch, not args.no_aug, 
                            args.no_val, args.data_root, args.cuda, args.seed, 
                            args.nworkers, args.dbg_ds_size, args.download)
    if args.adv_eval:
        adversarial_eval_dataset = get_adversarial_dataset(args)

    # Create activation function
    args.nonlin = args.nonlin.lower()
    if args.nonlin == "relu":
        nonlin = nn.ReLU
    elif args.nonlin == "step01":
        nonlin = partial(Step, make01=True, targetprop_rule=args.grad_tp_rule, 
                         scale_by_grad_out=not args.no_loss_grad_weight,
                         tanh_factor=args.softhinge_factor)
    elif args.nonlin == "step11":
        nonlin = partial(Step, make01=False, targetprop_rule=args.grad_tp_rule,
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
                           num_classes=num_classes, 
                           separate_activations=args.comb_opt, 
                           multi_gpu_modules=args.multi_gpu)
    elif args.model == "convnet8":
        network = ConvNet8(nonlin=nonlin, input_shape=input_shape,
                           num_classes=num_classes, 
                           separate_activations=args.comb_opt, 
                           multi_gpu_modules=args.multi_gpu)
    elif args.model == "toynet":
        if args.dataset not in ["cifar10", "mnist", "graphs"]:
            raise NotImplementedError(
                "Toy network can only be trained on CIFAR10, "
                "MNIST, or graph connectivity task.")
        network = ToyNet(nonlin=nonlin, input_shape=input_shape, 
            hidden_units=args.hidden_units, num_classes=num_classes, 
            biases=not args.no_biases, multi_gpu_modules=args.multi_gpu,
            separate_activations=args.comb_opt or args.collect_params)
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
    log_dir = os.path.join(os.path.join(os.curdir, "logs"), 
        str(round(time())) if not args.load_checkpoint else args.load_checkpoint)
    tb_logger = TensorBoardLogger(log_dir) if args.tb_logging else None

    print("Creating loss function...")
    criterion = torch.nn.CrossEntropyLoss(
        size_average=not args.comb_opt and not args.collect_params, 
        reduce=not args.comb_opt and not args.collect_params)

    if args.load_checkpoint is not None:
        print("Loading from checkpoint", args.load_checkpoint + "...")
        checkpoint_state = load_checkpoint(
            os.path.join(os.path.join(os.curdir, "logs"), args.load_checkpoint),
            best_eval=args.load_checkpoint_best)
        last_step = len(checkpoint_state["training_metrics"])-1
        model_state = checkpoint_state["model_state"]
        network.load_state_dict(model_state)
    if not args.no_train:
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
                    criterion=args.criterion, regions=3, 
                    perturb_scheduler=partial(perturb_scheduler, [1000]), 
                    candidates=args.candidates, iterations=args.iterations, 
                    searches=args.searches, search_type=args.search_type, 
                    perturb_type=args.perturb_type, candidate_type=args.candidate_type, 
                    candidate_grad_delay=args.candidate_grad_delay, 
                    splice_conv_targets=args.splice_conv_targets, 
                    grad_guide_offset=args.grad_guide_offset)
            elif args.comb_opt_method == "rand_grad":
                target_optimizer = partial(
                    RandomGradOptimizer, batch_size=args.batch, 
                    candidates=args.candidates, iterations=args.iterations, 
                    searches=args.searches)
            else:
                raise NotImplementedError
        else:
            target_optimizer = None

        # Train model
        with torch.cuda.device(args.device):
            try:
                last_step = train(network, train_loader, val_loader, criterion, optimizer, 
                    args.epochs, num_classes, target_optimizer=target_optimizer, 
                    log_dir=log_dir, logger=tb_logger, use_gpu=args.cuda, args=args)
            except KeyboardInterrupt:
                print("\nTraining interrupted!\n")
            else:
                print("\nFinished training.\n")

    if args.eval:
        print("Evaluating model...")
        eval_metrics = Metrics({"dataset_batches": len(val_loader), 
                                "loss": [], "accuracy": [], "accuracy_top5": []})
        with torch.no_grad():
            evaluate(network, val_loader, criterion, eval_metrics, 
                     tb_logger, last_step, log=True, use_gpu=args.cuda)

    if args.test:
        print("Testing model...")
        test_metrics = Metrics({"dataset_batches": len(test_loader), 
                                "loss": [], "accuracy": [], "accuracy_top5": []})
        with torch.no_grad():
            evaluate(network, test_loader, criterion, test_metrics, 
                     tb_logger, last_step, log=True, use_gpu=args.cuda)

    if args.ampl_eval:
        print("Running BARON evaluation...")
        print("Computing optimal targets at selected timesteps...")
        ampl_data_dir = args.ampl_eval_dir or os.path.join(log_dir, "run_data")
        for time_step in args.collect_timesteps:
            optimal_target_data = compute_optimal_targets(
                time_step, data_dir=ampl_data_dir, baron_options=args.baron_options, 
                model_file_path=("toy_model_batch_nobias.tex" 
                    if args.no_biases else "toy_model_batch.tex"))
            if optimal_target_data is not None:
                optimal_target_data = store_target_data(
                    optimal_target_data, ampl_data_dir)
                print("\nStep", time_step, "optimal target loss mean:", 
                      optimal_target_data["loss_mean"])
                print("Step", time_step, "optimal target loss standard error:", 
                      optimal_target_data["loss_std_error"])
                print("Step", time_step, "total time and ampl time:",
                      optimal_target_data["total_time"], 
                      optimal_target_data["ampl_time"])

    if args.adv_eval:
        print("\nEvaluating on adversarial examples...")
        if args.adv_attack == "all":
            attacks = adversarial.ATTACKS.keys()
        else:
            attacks = [args.adv_attack]
        for attack in attacks:
            with torch.cuda.device(args.device):
                print("Evaluating against attack", attack + "...")
                failure_rate = evaluate_adversarially(
                    network, adversarial_eval_dataset, "untargeted_misclassify", 
                    attack, args.test_batch, num_classes, 
                    args.adv_epsilon, args.cuda)
            print("Failure rate: %0.2f%%" % (100*failure_rate), "\n")

    if args.logs_to_dbx:
        upload(log_dir, token=args.dbx_token)


class Metrics(dict):

    def __len__(self):
        try:
            length = len(self[list(self.keys())[0]])
        except TypeError:
            length = len(self[list(self.keys())[-1]])
        return length

    def __getitem__(self, key):
        if isinstance(key, int):
            item = {metric: values[key] for metric, values in self.items() 
                    if isinstance(values, list)}
        else:
            item = super().__getitem__(key)
        return item

    def append(self, metric_tuple):
        for metric, value in metric_tuple:
            self[metric].append(value)


def train(model, train_dataset_loader, eval_dataset_loader, loss_function, 
          optimizer, epochs, num_classes, target_optimizer=None, 
          log_dir=None, logger=None, use_gpu=True, args=None):
    model.train()
    batch_size = train_dataset_loader.batch_size
    train_per_layer = target_optimizer is not None or args.collect_params
    if train_per_layer:
        optimizers = [optimizer(module.parameters()) for module in model.all_modules]
        modules = list(zip(model.all_modules, optimizers))[::-1]  # in reverse order
        lr_schedulers = [optim.lr_scheduler.MultiStepLR(optimizer, 
                            args.lr_decay_epochs, gamma=args.lr_decay_factor)
                         for optimizer in optimizers]
    else:
        optimizer = optimizer(model.parameters())
        lr_schedulers = [optim.lr_scheduler.MultiStepLR(
            optimizer, args.lr_decay_epochs, gamma=args.lr_decay_factor)]
    if target_optimizer is not None:
        target_optimizer = target_optimizer(
            modules=list(model.all_modules)[::-1],
            shapes=model.input_sizes[::-1],
            activations=list(model.all_activations[::-1]),
            loss_functions=[loss_function]*len(model.all_modules),
            use_gpu=use_gpu)
    print("Optimizers created.")
    extra_args = target_optimizer.__dict__.copy() if args.comb_opt else None
    if extra_args is not None and "perturb_base_tensors" in extra_args:
        del extra_args["perturb_base_tensors"]
    store_run_info(args, log_dir, extra_args=extra_args)
    training_metrics = Metrics({"dataset_batches": len(train_dataset_loader),
                                "loss": [], "accuracy": [], 
                                "accuracy_top5": [], "steps/sec": []})
    if eval_dataset_loader is not None:
        eval_metrics = Metrics({"dataset_batches": len(eval_dataset_loader), 
                                "loss": [], "accuracy": [], "accuracy_top5": []})
    else:
        eval_metrics = None
    print("\nBeginning training...\n")
    for epoch in range(epochs):
        for scheduler in lr_schedulers:
            scheduler.step()
        if epoch == 0 and eval_dataset_loader is not None:
            with torch.no_grad():
                evaluate(model, eval_dataset_loader, loss_function, 
                         eval_metrics, logger, 0, log=True, use_gpu=use_gpu)
        last_time, last_step = time(), 0
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
            if i % 10 == 1 or (args.ampl_train and 10 <= step < args.ampl_train_steps+10):
                last_step, last_time = eval_step(model, inputs, labels, loss_function, 
                                                 training_metrics, logger, epoch, 
                                                 i, step, last_time, last_step)
            if args.collect_params and target_optimizer is None:
                train_step_grad(model, modules, inputs, labels, loss_function, optimizer, 
                                target_optimizer, args, step, train_per_layer, log_dir)
            else:
                train_step(model, modules, inputs, labels, loss_function, optimizer, 
                           target_optimizer, args, step, train_per_layer, log_dir)
        if eval_dataset_loader is not None:
            with torch.no_grad():
                evaluate(model, eval_dataset_loader, loss_function, 
                         eval_metrics, logger, step, log=True, use_gpu=use_gpu)
        if args.store_checkpoints or args.store_many_checkpoints:
            store_checkpoint(model, optimizers if train_per_layer else optimizer, 
                             args, training_metrics, eval_metrics, epoch+1, log_dir, 
                             not args.store_many_checkpoints)
    return step


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
                if args.ampl_train and 10 <= step < args.ampl_train_steps+10:
                    data_strings = []
                    for i in range(args.batch):
                        data_string = parse_step_data(
                            model, targets[i], i, step, 
                            store_data=False, biases=not args.no_biases)
                        data_strings.append(data_string)
                    optimal_target_data, output = compute_optimal_targets(
                        step, data_strings=data_strings, 
                        baron_options=args.baron_options, 
                        model_file_path=("toy_model_batch_nobias.tex" 
                            if args.no_biases else "toy_model_batch.tex"))
                    valid_lengths = [args.hidden_units, args.hidden_units+1]
                    if all([target.shape[0] in valid_lengths 
                            for target in optimal_target_data["targets"]]):
                        optimal_target_data["targets"] = [
                            target.resize_(args.hidden_units) 
                            for target in optimal_target_data["targets"]]
                    else:
                        print([target.shape for target 
                               in optimal_target_data["targets"]])
                    try:
                        targets = torch.stack(optimal_target_data["targets"]).cuda()
                    except:
                        print("\n", output, "\n")
                        raise
                else:
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
            
    if args.collect_params and step in args.collect_timesteps and args.model == "toynet":
        target_loss = loss_function(modules[0][0](targets.float()), output_targets)
        for i in range(args.collect_count):
            parse_step_data(model, output_targets[i], i, step, 
                            os.path.join(log_dir, "run_data"), 
                            target_loss[i].item(), biases=not args.no_biases)
    if not train_per_layer:
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
    output_targets = targets
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
    
    if args.collect_params and step in args.collect_timesteps and args.model == "toynet":
        target_loss = loss_function(modules[0][0](targets.float()), output_targets)
        for i in range(args.collect_count):
            parse_step_data(model, output_targets[i], i, step, 
                            os.path.join(log_dir, "run_data"), 
                            target_loss[i].item(), biases=not args.no_biases)


def eval_step(model, inputs, labels, loss_function, training_metrics, 
              logger, epoch, batch, step, last_step_time, last_step):
    model.eval()
    outputs = model(inputs)
    loss = loss_function(outputs, labels).mean().item()
    batch_accuracy = accuracy(outputs, labels).item()
    if isinstance(model, ToyNet):
        batch_accuracy_top5 = 0
    else:
        batch_accuracy_top5 = accuracy_topk(outputs, labels, k=5).item()
    current_time = time()
    steps_per_sec = (step - last_step)/(current_time-last_step_time)
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
    return step, last_step_time


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
                     eval_metrics, epoch, log_dir, only_best_last=True):
    if eval_metrics is not None:
        best_index = max(enumerate(eval_metrics["accuracy"]), key=lambda val: val[1])[0]
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
    removed_epochs = []
    if (only_best_last and eval_metrics is not None 
            and best_index == epoch and epoch != 0):
        removed_epochs.append(max(enumerate(eval_metrics["accuracy"][:-1]), 
                                  key=lambda val: val[1])[0])
    if (only_best_last and (eval_metrics is None or epoch-1 != best_index) 
            and epoch-1 not in removed_epochs):
        removed_epochs.append(epoch-1)
    if not only_best_last and epoch > 9 and epoch % 10 != 0:
        removed_epochs.append(epoch-10)
    if removed_epochs:
        for epoch in removed_epochs:
            previous = os.path.join(log_dir, "checkpoint_epoch{}.state".format(epoch))
            if os.path.exists(previous) and os.path.isfile(previous):
                os.remove(previous)


def load_checkpoint(log_dir, epoch=None, best_eval=False):
    checkpoint_files = [file_name for file_name in os.listdir(log_dir)
                        if file_name.startswith("checkpoint")]
    if epoch is None and best_eval:
        checkpoint_states = [
            torch.load(os.path.join(log_dir, file_name), map_location="cpu") 
            for file_name in checkpoint_files]
        best_index, best_accuracy = 0, 0
        for i, state in enumerate(checkpoint_states):
            eval_accuracy = state["eval_metrics"][-1]["accuracy"]
            if eval_accuracy > best_accuracy:
                best_accuracy = eval_accuracy
                best_index = i
        checkpoint_files = [checkpoint_files[best_index]]
    elif epoch is None:
        checkpoint_files.sort(key=lambda string: int(string[16:-6]))
    else:
        checkpoint_files = [file_name for file_name in checkpoint_files
                            if "epoch{}".format(epoch) in file_name]

    checkpoint_state = torch.load(
        os.path.join(log_dir, checkpoint_files[-1]), map_location="cpu")
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


def parse_step_data(model, label, target_index, train_step, log_dir=None, 
                    target_loss=None, store_data=True, layer=2, biases=True):
    if log_dir is not None and not os.path.exists(log_dir):
        os.mkdir(log_dir)
    torch.set_printoptions(threshold=1000000, linewidth=1000000)
    parameters = list(model.parameters())
    weight_matrix = parameters[2*(layer-1) if biases else layer-1].transpose(0, 1)
    weight_rows = re.findall(r"\[.*?\]", str(weight_matrix))
    weight_rows = [correct_format(row, index=i+1) for i, row in enumerate(weight_rows)]
    if biases:
        bias_vector = parameters[2*(layer-1)+1]
        bias_rows = re.findall(r"\[.*?\]", str(bias_vector))
        bias_rows = [correct_format(row, index=i+1) for i, row in enumerate(bias_rows)]
    torch.set_printoptions(profile="default")

    if label.numel() == 1:
        label_vector = torch.zeros(10)
        label_vector[label] = 1
        label = label_vector
    label_rows = re.findall(r"\[.*?\]", str(label))
    label_rows = [correct_format(row, index=i+1) for i, row in enumerate(label_rows)]

    weight_string = "\n".join(weight_rows)
    if biases:
        bias_string = "\n".join(bias_rows)
    label_string = "\n".join(label_rows)

    data_string = "data ;\n\nparam M:=100 ;\nparam N:=10 ;\n\n\n"
    data_string += "param T:\n\t" + "\t".join(str(i) for i in range(1, 11)) + " :=\n"
    data_string += label_string + " ;\n\n\n"
    if biases:
        data_string += "param B:\n\t" + "\t".join(str(i) for i in range(1, 11)) + " :=\n"
        data_string += bias_string + " ;\n\n\n"
    data_string += "param W:\n\t\t" + "\t\t".join(str(i) for i in range(1, 11)) + " :=\n"
    data_string += weight_string + "  ;"

    if store_data:
        data_file_name = os.path.join(
            log_dir, "data" + str(target_index) + "_step" + str(train_step) + ".txt")
        with open(data_file_name, "w+") as data_file:
            print(data_string, file=data_file)
        loss_file_name = data_file_name.replace(
            "data" + str(target_index), "loss" + str(target_index))
        with open(loss_file_name, "w+") as loss_file:
            print(target_loss, file=loss_file)

    return data_string


def compute_optimal_targets(train_step, data_dir=None, data_strings=None, 
                            model_file_path="toy_model_batch.tex", 
                            target_index=None, baron_options=None):
    if data_strings is None:
        if target_index is not None:
            data_file_path = os.path.join(
                data_dir, "data" + str(target_index) + "_step" + str(train_step))
            data_file_paths = data_file_path
        else:
            data_file_names = utils.file_findall(
                data_dir, r"data\d+_step" + str(train_step) + r"\.txt")
            data_file_paths = [os.path.join(data_dir, file_name) 
                               for file_name in data_file_names]
        if not data_file_paths:
            print("\nNo AMPL data found for step", train_step)
            return
    else:
        data_file_paths = None
    optimal_target_data, output = run_neos_job(
        model_file_path, data_file_paths, data_strings, 
        display_variable_data=True, baron_options=baron_options, 
        batched_data=True)
    if optimal_target_data is not None:
        optimal_target_data["step"] = train_step
    return optimal_target_data, output


def store_target_data(target_data, data_dir):
    loss_mean = np.mean(target_data["losses"])
    loss_std_error = (np.std(target_data["losses"]) 
                      / np.sqrt(len(target_data["losses"])))
    target_data.update({"loss_mean": loss_mean, "loss_std_error": loss_std_error})
    file_path = os.path.join(
        data_dir, "ampl_output_step" + str(target_data["step"]) + ".data")
    torch.save(target_data, file_path)
    return target_data


if __name__ == "__main__":
    args = get_args()
    try:
        if args.runs != 1:
            multiprocessing.set_start_method("spawn")
    except RuntimeError:
        pass
    if args.load_checkpoints is not None:
            checkpoints = args.load_checkpoints
    else:
        checkpoints = [None]
    for checkpoint in checkpoints:
        args.load_checkpoint = checkpoint
        if args.runs == 1:
            main(args)
        else:
            for i in range(args.runs):
                print("\nStarting experiment " + str(i+1) + "...\n")
                # Run in separate process to avoid PyTorch multiprocessing errors
                process = Process(target=main, args=(args,))
                process.start()
                process.join()