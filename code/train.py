#!/usr/bin/env python3

import argparse
import logging
import os
import sys
import shutil
import warnings
from copy import deepcopy
from datetime import datetime
from functools import partial
from setproctitle import setproctitle
from socket import gethostname

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn

import targetprop as tp
#import targetpropoptimizer as tpo
from losses import multiclass_hinge_loss, multiclass_hinge_loss_softmax, multiclass_truncated_hinge_loss
from activations import Step, Staircase, OldStaircase, hardsigmoid, ThresholdReLU, CAbs
from datasets import create_datasets
#from sgd_line_search import SGDLineSearch
from util.tensorboardlogger import TensorBoardLogger
from util.timercontext import timer_context

from models.convnet4 import ConvNet4
from models.convnet8 import ConvNet8
from models.alexnet import AlexNet
from models.alexnet_dorefa import AlexNet as AlexNetDoReFa
from models.fcnet import FCNet
# from models.resnet import ResNet
from models.resnet_cifar import ResNet18
from models.resnet_imagenet import resnet18

from models.allconvnet import AllConvNet
from models.densenet_bamos import DenseNet
from models.hack_densenet import HackDenseNet

top1_model_name = 'best_model_top1.pth.tar'
top5_model_name = 'best_model_top5.pth.tar'


def main():
    # argument definitions
    parser = argparse.ArgumentParser()

    # model arguments
    parser.add_argument('--arch', type=str,
                        choices=('convnet4', 'convnet8', 'fcnet', 'fcnet0.5', 'fcnet0.1', 'allconvnet', 'resnet18',
                                 'densenet', 'shallowdensenet', 'hackdensenet', 'alexnet', 'alexnet_drf'),
                        help='model architecture to use')
    parser.add_argument('--nonlin', type=str,
                        choices=('step01', 'step11', 'relu', 'threshrelu', 'threshrelu2', 'hardtanh', 'cabs',
                                 'hardsigmoid', 'staircase', 'staircase3', 'staircase11_3', 'staircase_t2',
                                 'staircase7', 'oldstaircase7',
                                 'oldstaircase', 'staircase100', 'oldstaircase3', 'oldstaircase11', 'oldstaircase11_3'),
                        help='non-linearity to use in the specified architecture')
    parser.add_argument('--loss', type=str, default='hinge',
                        choices=('crossent', 'hinge', 'hingeL2', 'hingeSM', 'hingeTrunc'),
                        help='the loss function to use for training')
    parser.add_argument('--use-bn', action='store_true', default=False,
                        help='if specified, use batch-normed version of the current architecture')
    parser.add_argument('--no-step-last', action='store_true', default=False,
                        help='if set, the last layer''s step function is replaced with a model-dependent '
                             'high-precision non-linearity')
    # parser.add_argument('--winit', type=str, default='default',
    #                     choices=('xavier', ''),
    #                     help='weight initialization method to use')

    parser.add_argument('--test-model', type=str,
                        help='specify a filename to a pre-trained model, which will then be evaluated on the test '
                             'set of the specified dataset')

    # targetprop arguments
    parser.add_argument('--tp-rule', type=str, default='TruncWtHinge',
                        choices=tuple([e.name for e in tp.TPRule]),
                        help='the TargetProp rule to use')
    # parser.add_argument('--tp-heur', type=str, default='SignGrad',
    #                     choices=tuple(e.name for e in tpo.TargetHeuristics),
    #                     help='the heuristic to use for setting targets')
    parser.add_argument('--tp-momentum', type=float, default=0.0,
                        help='momentum amount for TargetProp')
    parser.add_argument('--tp-updates', type=int, default=0,
                        help='number of weight updates per step-function block during targetprop')
    parser.add_argument('--tp-greedy', action='store_true', default=False)
    parser.add_argument('--tp-grad-scale', action='store_true', default=False,
                        help='re-scale each layer''s loss by the gradient of its downstream layer')
    parser.add_argument('--no-tp-grad-scale', action='store_true', default=False,
                        help='do not re-scale each layer''s loss by the gradient of its downstream layer')

    # optimizer arguments
    parser.add_argument('--batch', type=int, default=64,
                        help='batch size to use for training')
    parser.add_argument('--epochs', type=int, default=300,
                        help='number of epochs to train for')
    parser.add_argument('--opt', type=str, default='sgd',
                        choices=('sgd', 'adam', 'rmsprop', 'sgd_ls'),
                        help='optimizer to use to train')
    parser.add_argument('--lr', type=float,
                        help='starting learning rate')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='momentum amount for SGD')
    parser.add_argument('--wtdecay', type=float, default=0,
                        help='weight decay (L2 regularization) amount')
    parser.add_argument('--lr-decay', type=float, default=1.0,
                        help='factor by which to multiply the learning rate at each value in <lr-decay-epochs>')
    parser.add_argument('--lr-decay-epochs', type=int, nargs='+', default=None,
                        help='list of epochs at which to multiply the learning rate by <lr-decay>')
    parser.add_argument('--test-batch', type=int, default=0,
                        help='batch size to use for validation and testing')

    parser.add_argument('--no-tpo', action='store_true', default=False)

    # dataset arguments
    parser.add_argument('--ds', type=str, default='cifar10',
                        choices=('mnist', 'cifar10', 'cifar100', 'svhn', 'imagenet'),
                        help='dataset on which to train')
    parser.add_argument('--data-root', type=str, default='',
                        help='root directory for imagenet dataset (with separate train, val, test folders)')
    # parser.add_argument('--use-test', action='store_true', default=False,
    #                     help='if specified, evaluate on test data instead of validation data')
    parser.add_argument('--no-val', action='store_true', default=False,
                        help='if specified, do not create a validation set from the training data and '
                             'use it to choose the best model')
    parser.add_argument('--plot-test', action='store_true', default=False,
                        help='if specified, plot test performance per training epoch')
    parser.add_argument('--test-final-model', action='store_true', default=False,
                        help='if specified, evaluate the final trained model on the test set')
    parser.add_argument('--no-aug', action='store_true',
                        help='if specified, do not use data augmentation (default=True for MNIST, False for CIFAR10)')
    parser.add_argument('--download', action='store_true',
                        help='allow downloading of the dataset (not including imagenet) if not found')
    parser.add_argument('--dbg-ds-size', type=int, default=0,
                        help='debug: artificially limit the size of the training data')

    # logging arguments
    parser.add_argument('--save', type=str,
                        help='directory in which to save output')
    parser.add_argument('--no-save', action='store_true', default=False,
                        help='if specified, don''t log or save anything about this run to disk')
    parser.add_argument('--log-info', type=str, default='',
                        help='info to append to the name of the log file')

    # tensorboard arguments
    parser.add_argument('--no-tb-log', action='store_true', default=False,
                        help='if specified, don''t log output to tensorboard')
    parser.add_argument('--tb-sample-pct', type=float, default=0.01,
                        help='percentage to subsample data for tensorboard histograms')
    parser.add_argument('--tb-drop-pct', type=float, default=0.95,
                        help='percentage of batches to not include in tensorboard histograms')
    parser.add_argument('--diagnostics', action='store_true', default=False,
                        help='if specified, log extra diagnostic information to tensorboard')

    # other arguments
    parser.add_argument('--resume', type=str, metavar='PATH',
                        help='path to checkpoint to resume training from (default: none)')
    parser.add_argument('--gpus', type=int, default=[0], nargs='+',
                        help='which GPU device ID(s) to train on')
    # parser.add_argument('--ngpus', type=int, default=1,
    #                     help='number of GPUs to train on (ngpus > 1 is only usable if --no-tpo is set)')
    parser.add_argument('--nworkers', type=int, default=2,
                        help='number of workers to use for loading data from disk')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='if specified, use CPU only')
    parser.add_argument('--seed', type=int, default=468412397,
                        help='random seed')

    arg_list = deepcopy(sys.argv)
    args = parser.parse_args()

    if args.arch is None or args.nonlin is None or args.lr is None:
        print('ERROR: arch, nonlin, and lr arguments must be specified\n')
        # parser.print_usage()
        parser.print_help()
        exit(-1)

    uses_tp = args.nonlin.startswith('step') or args.nonlin.startswith('stair') or args.nonlin.startswith('oldstair')

    if uses_tp and ((args.tp_grad_scale and args.no_tp_grad_scale)
                    or not (args.tp_grad_scale or args.no_tp_grad_scale)):
        print('ERROR: exactly one of tp-grad-scale and no-tp-grad-scale must be specified with {}'.format(args.nonlin))
        parser.print_help()
        exit(-1)

    if len(args.gpus) > 1 and not args.no_tpo:
        print('ERROR: cannot use multiple GPUs with TPO')
        exit(-1)

    curtime = datetime.now().strftime('%Y.%m.%d_%H.%M.%S')

    gpu_str = ','.join(str(g) for g in args.gpus)
    if not args.no_cuda:
        os.environ['CUDA_VISIBLE_DEVICES'] = gpu_str  # must be set before torch.cuda is called

    args.cuda = not args.no_cuda and torch.cuda.is_available()

    arch_str = args.arch + ('BN' if args.use_bn else '')
    op_str = args.opt + '_lr{}_mu{}_wd{}{}'.format(args.lr, args.momentum, args.wtdecay, 'noval' if args.no_val else '')
    # tp_str = args.nonlin + ('-' + args.tp_rule if uses_tp else '') + ('-' + args.tp_heur if uses_tp else '')
    tp_str = args.nonlin + ('-' + args.tp_rule if uses_tp else '') + '-'
    tp_str = tp_str + ('-ngs' if args.no_tp_grad_scale else '') + ('-nsl' if args.no_step_last else '')
    setproctitle(args.save or args.ds + '.' + arch_str + '.' + op_str + '.' + tp_str)
    args.save = args.save or os.path.join('logs', args.ds, curtime+'.'+arch_str+'.'+args.loss+'.'+op_str+'.'+tp_str)
    if args.log_info:
        args.save = args.save + '.' + args.log_info
    if args.test_model:
        args.save = args.save + '_test'

    args.tp_rule = tp.TPRule[args.tp_rule]
    # args.tp_heur = tpo.TargetHeuristics[args.tp_heur]
    if not args.no_aug:
        args.no_aug = True if args.ds == 'mnist' else False
    args.no_save = False if not args.no_save else args.no_save
    args.tp_greedy = False if not args.tp_greedy else args.tp_greedy
    args.no_step_last = False if not args.no_step_last else args.no_step_last
    args.download = False if not args.download else args.download
    if not args.tp_grad_scale:
        # assert args.no_tp_grad_scale
        args.tp_grad_scale = False
    if args.test_model:
        args.no_val = True

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if args.cuda:
        print('using cuda device{}: {}'.format('s' if len(args.gpus) > 1 else '', gpu_str))
        # torch.cuda.set_device(args.gpu)
        torch.cuda.manual_seed(args.seed)

    if not args.no_save:
        if os.path.exists(args.save):
            shutil.rmtree(args.save)
        os.makedirs(args.save, exist_ok=True)

    log_formatter = logging.Formatter('%(asctime)s.%(msecs)03d [%(levelname)-5.5s] %(message)s',
                                      datefmt='%Y.%m.%d %H:%M:%S')
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)

    if not args.no_save:
        print("logging to file '{}.log'".format(args.save))
        file_handler = logging.FileHandler(args.save+'.log')
        file_handler.setFormatter(log_formatter)
        file_handler.setLevel(logging.DEBUG)
        root_logger.addHandler(file_handler)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(log_formatter)
    # console_handler.setLevel(logging.INFO)
    console_handler.setLevel(logging.DEBUG)
    root_logger.addHandler(console_handler)

    logging.getLogger('PIL.Image').setLevel(logging.WARNING)
    logging.getLogger('PIL').setLevel(logging.WARNING)
    logging.getLogger('tensorflow').setLevel(logging.WARNING)

    logging.info('hostname: {}'.format(gethostname()))
    logging.info('command line call: {}'.format(" ".join(arg_list)))
    logging.info('arguments: {}'.format(args))

    if not args.test_model:
        logging.info("training a deep network with the specified arguments")
    else:
        logging.info("testing model '{}' with the specified arguments".format(args.test_model))

    torch.utils.backcompat.broadcast_warning.enabled = True  # enable warnings for possible broadcasting errors
    warnings.filterwarnings("ignore", "Corrupt EXIF data")   # ignore warnings for corrupt EXIF data (imagenet)
    warnings.filterwarnings("ignore", "Possibly corrupt EXIF data")

    # with torch.cuda.device(args.gpu if args.cuda else -1):

    # ----- create datasets -----
    train_loader, val_loader, test_loader, num_classes = \
        create_datasets(args.ds, args.batch, args.test_batch, not args.no_aug, args.no_val, args.data_root,
                        args.cuda, args.seed, args.nworkers, args.dbg_ds_size, args.download)

    use_top5 = (args.ds.lower() == 'imagenet')

    metrics = {'loss': Metric('loss', float('inf'), False),
               'acc1': Metric('acc1', 0.0, True),
               'acc5': Metric('acc5', 0.0, True)}
    metrics = {'train': deepcopy(metrics), 'val': deepcopy(metrics), 'test': deepcopy(metrics)}

    # ----- create loss function -----
    loss_function = get_loss_function(args.loss)

    # either train a model from scratch or load and evaluate a model from disk
    if not args.test_model:
        # ----- create model -----
        model = create_model(args, num_classes)

        tb_logger = TensorBoardLogger(args.save) if not args.no_save and not args.no_tb_log else None
        logging.info('created {} model:\n {}'.format(arch_str, model))
        logging.info("{} model has {} parameters".format(arch_str,
                                                         sum([p.data.nelement() for p in model.parameters()])))
        print('num params: ', [p.data.nelement() for p in model.parameters()])

        # ----- create optimizer -----
        opt_args = {'lr': args.lr, 'weight_decay': args.wtdecay}
        if args.opt == 'sgd':
            optimizer = optim.SGD(model.parameters(), lr=args.lr,
                                  momentum=args.momentum, weight_decay=args.wtdecay)
            opt = optim.SGD
            opt_args['momentum'] = args.momentum
        elif args.opt == 'adam':
            optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wtdecay)
            opt = optim.Adam
        elif args.opt == 'rmsprop':
            optimizer = optim.RMSprop(model.parameters(), lr=args.lr, weight_decay=args.wtdecay)
            opt = optim.RMSprop
        # elif args.opt == 'sgd_ls':
        #     optimizer = SGDLineSearch(model.parameters(), lr=args.lr, weight_decay=args.wtdecay)
        #     opt = SGDLineSearch
        else:
            raise NotImplementedError('no other optimizers currently implemented')

        if not args.no_tpo:
            ds_size = len(train_loader)*args.batch
            # optimizer = tpo.TargetPropOptimizer(model, ds_size, opt, opt_args,
            #                                     tp_rule=args.tp_rule,
            #                                     tp_heur=args.tp_heur,
            #                                     tp_momentum=args.tp_momentum,
            #                                     tp_updates=args.tp_updates,
            #                                     tp_greedy=args.tp_greedy,
            #                                     tp_nesterov=False,
            #                                     tp_grad_scale=args.tp_grad_scale)

        # ----- load checkpoint if specified -----
        start_epoch, timers, best_acc_top1, best_acc_top5 = 1, {}, 0.0, 0.0
        if args.resume:
            if os.path.isfile(args.resume):
                logging.info("loading state from training checkpoint '{}'".format(args.resume))
                checkpoint = torch.load(args.resume)
                if args.arch != checkpoint['arch']:
                    logging.warning("loaded checkpoint has different arch ({}) than current arch ({})"
                                    .format(checkpoint['arch'], args.arch))
                if args.loss != checkpoint['loss']:
                    logging.warning("loaded checkpoint has different loss function ({}) than currently in use ({})"
                                    .format(checkpoint['loss'], args.loss))
                if args.nonlin != checkpoint['nonlin']:
                    logging.warning("loaded checkpoint has different non-linearity ({}) than currently in use ({})"
                                    .format(checkpoint['nonlin'], args.nonlin))
                start_epoch = checkpoint['epoch'] + 1
                model.load_state_dict(checkpoint['model_state'])
                optimizer.load_state_dict(checkpoint['opt_state'])
                if 'metric_state' in checkpoint:
                    metrics = checkpoint['metric_state']
                else:
                    metrics['val']['acc1'].val = checkpoint['best_val_acc_top1']
                    metrics['val']['acc5'].val = checkpoint['best_val_acc_top5']
                timers = checkpoint['timers']
                logging.info("resuming training from checkpoint '{}' at epoch {}".format(args.resume, start_epoch))
            else:
                logging.error("no model checkpoint found at '{}', exiting".format(args.resume))
                exit(-1)

        if val_loader:
            logging.info('evaluating training on validation data (train size = {}, val size = {}, test size = {})'
                         .format(len(train_loader.dataset), len(val_loader.dataset), len(test_loader.dataset)))
        else:
            logging.info('not using validation data (train size = {}, test size = {})'
                         .format(len(train_loader.dataset), len(test_loader.dataset)))

        if tb_logger is not None and args.diagnostics:
            register_step_tb_hooks(model, tb_logger, args.tb_drop_pct, args.tb_sample_pct)

        cudnn.benchmark = True

        # ----- train the model -----
        best_model_state = train_model(start_epoch, args.epochs, model, optimizer, loss_function,
                                       model.needs_backward_twice, tb_logger, train_loader,
                                       None if args.no_val else val_loader,
                                       test_loader if args.plot_test else None,
                                       args.cuda, args.lr_decay, args.lr_decay_epochs,
                                       args.save if not args.no_save else '',
                                       use_top5, metrics, args, timers)

        if args.test_final_model:
            logging.info('testing on trained model ({})'.format('final' if args.no_val
                                                                else ('top-5' if use_top5 else 'top-1')))
            model.load_state_dict(best_model_state)
            test_model(model, loss_function, test_loader, args.cuda, True, use_top5)

    else:
        model = create_model(args, num_classes)
        logging.info("loading test model from '{}'".format(args.test_model))
        state = torch.load(args.test_model)
        model.load_state_dict(state['model_state'])
        test_model(model, loss_function, test_loader, args.cuda, True, use_top5)

    if not args.no_save:
        print('')
        logging.info("log file: '{}.log'".format(args.save))
        logging.info("log dir: '{}'".format(args.save))
        if not args.test_model:
            logging.info("best top-1 accuracy model: '{}'".format(os.path.join(args.save, top1_model_name)))
            logging.info("best top-5 accuracy model: '{}'".format(os.path.join(args.save, top5_model_name)))


def train_model(start_epoch, num_epochs, model, optimizer, loss_function, needs_backward_twice, tb_logger,
                train_loader, val_loader, test_loader, use_cuda, lr_decay, lr_decay_epochs, log_dir, use_top5,
                metrics, args, timers):

    epoch = start_epoch
    best_model_state = None
    timer = partial(timer_context, timers_dict=timers)

    # @profile
    def train_epoch(epoch):
        model.train()
        ds_size = len(train_loader.dataset)
        num_batches = int(np.ceil(float(ds_size) / train_loader.batch_size))

        for batch_idx, (data, target, index) in enumerate(train_loader):
            it = epoch * int(num_batches) + batch_idx
            if use_cuda:
                with timer('cuda'):
                    data, target = data.cuda(), target.cuda()
            data, target = Variable(data), Variable(target)

            # if isinstance(optimizer, tpo.TargetPropOptimizer):
            #     with timer('forward_backward'):
            #         loss, output = optimizer.forward_backward(epoch, it, data, target, index, loss_function,
            #                                                   timers, tb_logger)
            #         lossf = loss.squeeze()[0]
            # else:
            with timer('forward'):
                optimizer.zero_grad()
                output = model(data)
            with timer('loss'):
                loss = loss_function(output, target)
                lossf = loss.data.squeeze()[0]
                output = output.data
            with timer('backward'):
                loss.backward(retain_graph=needs_backward_twice)
                if needs_backward_twice:
                    assert False
                    optimizer.zero_grad()
                    loss.backward()
            with timer('optimizer'):
                optimizer.step()

            with timer('output'):
                top_pred = output.max(dim=1)[1]
                top1_acc = 100.0 * top_pred.eq(target.data).float().cpu().mean()
                _, top5_pred = torch.topk(output, 5, dim=1, largest=True)
                assert top5_pred.size(0) == target.size(0) and target.dim() == 1
                top5_acc = 100.0 * (top5_pred == target.data.unsqueeze(1)).max(dim=1)[0].float().cpu().mean()

                metrics['train']['loss'].update(lossf, epoch)
                metrics['train']['acc1'].update(top1_acc, epoch)
                metrics['train']['acc5'].update(top5_acc, epoch)

            with timer('output'):
                if num_batches <= 5 or (batch_idx % (num_batches // 5)) == 0:
                    param_norms = torch.FloatTensor([p.data.view(-1).norm(p=2) / p.numel()
                                                     for p in model.parameters() if p.data is not None])
                    grad_norms = torch.FloatTensor([p.grad.data.view(-1).norm(p=2) / p.grad.numel()
                                                    for p in model.parameters() if p.grad is not None])
                    logging.info("Train epoch {} [{}/{} ({:.0f}%)]:\t loss = {:.6f}, top1 accuracy = {:.2f}, "
                                 "{}param norms: {}, grad norms: {}"
                                 .format(epoch, (batch_idx+1) * len(data), ds_size,
                                         100. * (batch_idx+1) / len(train_loader), lossf, top1_acc,
                                         "top5 accuracy = {:.2f}, ".format(top5_acc) if use_top5 else "",
                                         torch.topk(param_norms, 2)[0].tolist(), torch.topk(grad_norms, 2)[0].tolist()))

            with timer('tensor_board'):
                if tb_logger is not None:
                    if it % 200 == 0:
                        tb_logger.scalar_summary('train/loss', lossf, it)
                        tb_logger.scalar_summary('train/top1_accuracy', top1_acc, it)
                        tb_logger.scalar_summary('train/top5_accuracy', top5_acc, it)
                        tb_logger.log_stored_histograms(it, prefix='train/')
                    else:
                        tb_logger.clear_histograms()

    # @profile
    def test(epoch, data_loader, is_val):
        ds_size = len(data_loader.dataset)
        test_loss, top1_acc, top5_acc = test_model(model, loss_function, data_loader, use_cuda, False)
        with timer('output'):
            if is_val:
                print('')
            log_str = '{} set: average loss = {:.4f}, top1 accuracy = {}/{} ({:.2f}%)'\
                .format('Validation' if is_val else 'Test', test_loss, round(top1_acc*ds_size/100), ds_size, top1_acc)
            log_str += (', top5 accuracy = {}/{} ({:.2f}%)'.format(round(top5_acc*ds_size/100), ds_size, top5_acc)
                        if use_top5 else '')
            logging.info(log_str)
            print('')

        ds_name = 'val' if is_val else 'test'
        metrics[ds_name]['loss'].update(test_loss, epoch)
        is_best1 = metrics[ds_name]['acc1'].update(top1_acc, epoch)
        is_best5 = metrics[ds_name]['acc5'].update(top5_acc, epoch)
        nonlocal best_model_state
        if is_val and ((not use_top5 and is_best1) or (use_top5 and is_best5)):
            best_model_state = deepcopy(model.state_dict())

        with timer('tensor_board'):
            if tb_logger is not None:
                tb_logger.scalar_summary(ds_name+'/loss', test_loss, epoch)
                tb_logger.scalar_summary(ds_name+'/top1_accuracy', top1_acc, epoch)
                tb_logger.scalar_summary(ds_name+'/top5_accuracy', top5_acc, epoch)
                tb_logger.log_stored_histograms(epoch, prefix=ds_name+'/')
        return is_best1, is_best5

    try:
        # run the train + test loop for <num_epochs> iterations
        for epoch in range(start_epoch, num_epochs+1):
            is_best_top1, is_best_top5 = False, False
            exp_lr_scheduler(optimizer, epoch, lr_decay=lr_decay, lr_decay_epoch=lr_decay_epochs)

            with timer('train'):
                train_epoch(epoch)

            if tb_logger is not None:
                tb_logger.clear_histograms()

            if val_loader is not None:
                with timer('val'):
                    is_best_top1, is_best_top5 = test(epoch, val_loader, True)

            if test_loader is not None:
                with timer('test'):
                    test(epoch, test_loader, False)

            if epoch % 25 == 0 or epoch == num_epochs:
                logging.info('timings: {}'.format(', '.join('{}: {:.3f}s'.format(*tt) for tt in
                                                            zip(timers.keys(), timers.values()))))

            checkpoint_model(epoch, model=model, opt=optimizer, args=args, log_dir=log_dir,
                             metrics=metrics, timers=timers, is_best_top1=is_best_top1, is_best_top5=is_best_top5)

    except KeyboardInterrupt:
        print('KeyboardInterrupt: shutdown requested ... exiting')
        sys.exit(0)

    finally:
        for metric_name in ['acc1'] + (['acc5'] if use_top5 else []):
            for ds_name in ['train'] + (['val'] if val_loader is not None else []) + \
                    (['test'] if test_loader is not None else []):
                logging.info('best {} accuracy ({}): {:.2f}% occurred on epoch {} / {}'
                             .format(ds_name, 'top-1' if metric_name == 'acc1' else 'top-5',
                                     metrics[ds_name][metric_name].val, metrics[ds_name][metric_name].tag, epoch))

        logging.info('timings: {}'.format(', '.join('{}: {:.3f}s'.format(*tt) for tt in
                                                    zip(timers.keys(), timers.values()))))

    # if no validation set, then just use the final model
    if best_model_state is None:
        best_model_state = model.state_dict()

    return best_model_state


def test_model(model, loss_function, data_loader, use_cuda, log_results=True, print_top5=True):
    model.eval()
    loss, num_correct1, num_correct5, nsamples = 0, 0, 0, len(data_loader.dataset)
    for data, target in data_loader:
        if use_cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data, volatile=True), Variable(target, volatile=True)
        output = model(data)
        batch_loss = torch.squeeze(loss_function(output, target)).data[0]
        loss += batch_loss * data.size(0)
        output, target = output.data, target.data
        num_correct1 += output.max(1)[1].eq(target).float().cpu().sum()
        num_correct5 += (torch.topk(output, 5, dim=1)[1] == target.unsqueeze(1)).max(dim=1)[0].float().cpu().sum()
    test_top1_acc = 100. * num_correct1 / nsamples
    test_top5_acc = 100. * num_correct5 / nsamples
    loss /= nsamples

    if log_results:
        log_str = '\nTest set: average loss = {:.4f}, top1 accuracy = {}/{} ({:.2f}%)'\
            .format(loss, num_correct1, nsamples, test_top1_acc)
        log_str += (', top5 accuracy = {}/{} ({:.2f}%)\n'.format(num_correct5, nsamples, test_top5_acc)
                    if print_top5 else '\n')
        logging.info(log_str)

    return loss, test_top1_acc, test_top5_acc


def exp_lr_scheduler(optimizer, cur_epoch, lr_decay=0.1, lr_decay_epoch=7):
    """Decay learning rate by a factor of <lr_decay> every <lr_decay_epoch> epochs"""
    if ((isinstance(lr_decay_epoch, int) and cur_epoch % lr_decay_epoch == 0) or
            (isinstance(lr_decay_epoch, list) and cur_epoch in lr_decay_epoch)):
        logging.info('decaying learning rate by a factor of %g' % lr_decay)
        sub_optimizers = [optimizer]
        for opt in sub_optimizers:
            for param_group in opt.param_groups:
                param_group['lr'] *= lr_decay
    return optimizer


def get_loss_function(loss_str):
    if loss_str == 'hinge':
        loss_function = multiclass_hinge_loss
    elif loss_str == 'crossent':
        loss_function = F.cross_entropy
    elif loss_str == 'hingeSM':
        loss_function = multiclass_hinge_loss_softmax
    elif loss_str == 'hingeTrunc':
        loss_function = multiclass_truncated_hinge_loss
    else:
        raise NotImplementedError('no other loss functions currently implemented')
    return loss_function


def create_model(args, num_classes):

    # determine the correct non-linearity
    nl_str = args.nonlin.lower()
    if nl_str == 'relu':
        nonlin = nn.ReLU
    elif nl_str == 'hardtanh':
        nonlin = nn.Hardtanh
    elif nl_str == 'hardsigmoid':
        nonlin = hardsigmoid
        assert False
    elif nl_str == 'cabs':
        nonlin = CAbs
    elif nl_str == 'threshrelu':
        nonlin = ThresholdReLU
    elif nl_str == 'threshrelu2':
        nonlin = partial(ThresholdReLU, max_val=2)
    elif nl_str == 'step01':
        nonlin = partial(Step, targetprop_rule=args.tp_rule, make01=True, scale_by_grad_out=args.tp_grad_scale)
    elif nl_str == 'step11':
        nonlin = partial(Step, targetprop_rule=args.tp_rule, make01=False, scale_by_grad_out=args.tp_grad_scale)
    elif nl_str == 'staircase':
        nonlin = partial(Staircase, targetprop_rule=args.tp_rule, nsteps=5, margin=1, trunc_thresh=2,
                         scale_by_grad_out=args.tp_grad_scale)
    elif nl_str == 'staircase3':
        nonlin = partial(Staircase, targetprop_rule=args.tp_rule, nsteps=3, margin=1, trunc_thresh=2,
                         scale_by_grad_out=args.tp_grad_scale)
    elif nl_str == 'staircase11_3':
        nonlin = partial(Staircase, targetprop_rule=args.tp_rule, nsteps=3, margin=1, trunc_thresh=2, a=-1,
                         scale_by_grad_out=args.tp_grad_scale)
    elif nl_str == 'staircase100':
        nonlin = partial(Staircase, targetprop_rule=args.tp_rule, nsteps=100, margin=1, trunc_thresh=2,
                         scale_by_grad_out=args.tp_grad_scale)
    elif nl_str == 'staircase_t2':
        nonlin = partial(Staircase, targetprop_rule=args.tp_rule, nsteps=5, margin=1, trunc_thresh=3,
                         scale_by_grad_out=args.tp_grad_scale)
    elif nl_str == 'staircase7':
        nonlin = partial(Staircase, targetprop_rule=args.tp_rule, nsteps=7, margin=1, trunc_thresh=2,
                         scale_by_grad_out=args.tp_grad_scale)
    elif nl_str == 'oldstaircase':
        nonlin = partial(OldStaircase, targetprop_rule=args.tp_rule, scale_by_grad_out=args.tp_grad_scale)
    elif nl_str == 'oldstaircase3':
        nonlin = partial(OldStaircase, targetprop_rule=args.tp_rule, nsteps=3, scale_by_grad_out=args.tp_grad_scale)
    elif nl_str == 'oldstaircase11':
        nonlin = partial(OldStaircase, targetprop_rule=args.tp_rule, a=-1, scale_by_grad_out=args.tp_grad_scale)
    elif nl_str == 'oldstaircase11_3':
        nonlin = partial(OldStaircase, targetprop_rule=args.tp_rule, a=-1, nsteps=3,
                         scale_by_grad_out=args.tp_grad_scale)
    elif nl_str == 'oldstaircase7':
        nonlin = partial(OldStaircase, targetprop_rule=args.tp_rule, nsteps=7, scale_by_grad_out=args.tp_grad_scale)
    else:
        raise NotImplementedError('no other non-linearities currently supported')

    if args.ds == 'mnist':
        input_shape = (1, 28, 28)
    elif args.ds.startswith('cifar'):
        input_shape = (3, 32, 32)
    elif args.ds == 'svhn':
        input_shape = (3, 40, 40)
    elif args.ds == 'imagenet':
        input_shape = (3, 224, 224)
    else:
        raise NotImplementedError('no other datasets currently supported')

    # create a model with the specified architecture
    if args.arch == 'convnet4':
        model = ConvNet4(nonlin=nonlin, use_bn=args.use_bn, input_shape=input_shape)
        assert args.ds == 'mnist' or args.ds == 'cifar10'
    elif args.arch == 'allconvnet':
        model = AllConvNet(nonlin=nonlin, use_bn=args.use_bn)
    elif args.arch == 'densenet':
        model = DenseNet(growthRate=12, depth=100, reduction=0.5, bottleneck=True, nClasses=num_classes,
                         nonlin=nonlin, useBN=args.use_bn)
    elif args.arch == 'shallowdensenet':
        # model = DenseNet(growthRate=12, depth=5, reduction=0.5, bottleneck=True, nClasses=num_classes,
        model = DenseNet(growthRate=12, depth=10, reduction=0.5, bottleneck=False, nClasses=num_classes,
                        nonlin=nonlin, useBN=args.use_bn)
    elif args.arch == 'hackdensenet':
        model = HackDenseNet(growthRate=12, depth=7, reduction=0.5, bottleneck=False, nClasses=num_classes,
                nonlin=nonlin, useBN=args.use_bn)
    elif args.arch == 'alexnet':
        model = AlexNet(nonlin=nonlin, no_step_last=args.no_step_last, num_classes=num_classes)
    elif args.arch == 'alexnet_drf':
        assert args.use_bn
        model = AlexNetDoReFa(nonlin=nonlin, no_step_last=args.no_step_last, use_bn=args.use_bn,
                              num_classes=num_classes, data_parallel=len(args.gpus) > 1)
    elif args.arch == 'convnet8':
        model = ConvNet8(nonlin=nonlin, use_bn=args.use_bn, input_shape=input_shape, no_step_last=args.no_step_last)
    elif args.arch == 'fcnet':
        model = FCNet(nonlin=nonlin, input_shape=input_shape, filter_frac=1.0)
    elif args.arch == 'fcnet0.5':
        model = FCNet(nonlin=nonlin, input_shape=input_shape, filter_frac=0.5)
    elif args.arch == 'fcnet0.1':
        model = FCNet(nonlin=nonlin, input_shape=input_shape, filter_frac=0.1)
    elif args.arch == 'resnet18':
        # model = ResNet(3, nonlin=nonlin, use_bn=args.use_bn)
        assert args.use_bn, 'batchnorm is currently required for resnet'
        if args.ds.startswith('cifar'):
            model = ResNet18(nonlin=nonlin)
        else:
            model = resnet18(False, nonlin=nonlin)
    else:
        raise NotImplementedError('other models not yet supported')

    logging.info("{} model has {} parameters and non-linearity={} ({})"
                 .format(args.arch, sum([p.data.nelement() for p in model.parameters()]), nl_str, args.tp_rule.name))

    if len(args.gpus) > 1 and args.arch != 'alexnet_drf':
        model = nn.DataParallel(model)

    if args.cuda:
        model.cuda()

    model.needs_backward_twice = False
    if nl_str.startswith('step') or nl_str == 'staircase':
        model.targetprop_rule = args.tp_rule
        model.needs_backward_twice = tp.needs_backward_twice(args.tp_rule)

    return model


class Metric:
    def __init__(self, name, init_val, want_max):
        self.name = name
        self.val = init_val
        self.tag = None
        self.want_max = want_max

    def update(self, val, tag):
        updated = False
        if (self.want_max and val > self.val) or (not self.want_max and val < self.val):
            self.val, self.tag, updated = val, tag, True
        return updated


def checkpoint_model(epoch, model, opt, args, log_dir, metrics, timers, is_best_top1, is_best_top5):
    if not log_dir:
        return

    state = {
        'epoch': epoch,
        'arch': args.arch,
        'nonlin': args.nonlin,
        'loss': args.loss,
        'model_state': model.state_dict(),
        'opt_state': opt.state_dict(),
        'metric_state': metrics,
        'args': args,
        'timers': timers,
    }

    # checkpoint full training state
    cp_name = 'model_checkpoint_epoch{}.pth.tar'
    top1_file = os.path.join(log_dir, top1_model_name)
    top5_file = os.path.join(log_dir, top5_model_name)
    torch.save(state, os.path.join(log_dir, cp_name.format(epoch)))
    if epoch > 1:
        prev_cp = os.path.join(log_dir, cp_name.format(epoch - 1))
        if os.path.exists(prev_cp) and os.path.isfile(prev_cp):
            os.remove(prev_cp)
    else:
        logging.info("model checkpoints will be saved to file '{}' after each epoch".format(cp_name.format('<epoch>')))
        logging.info("the best top-1 accuracy model will be saved to '{}'".format(top1_file))
        logging.info("the best top-5 accuracy model will be saved to '{}'".format(top5_file))

    # save model state for best top-1 and top-5 models
    del state['opt_state']
    if is_best_top1:
        torch.save(state, top1_file)
    if is_best_top5:
        torch.save(state, top5_file)


def register_step_tb_hooks(model, tb_logger, drop_percent=0.99, sample_percent=0.05):
    def forward_tb_hook(_, input, output, input_hook=None, output_hook=None):
        assert len(input) == 1
        input_hook(input[0].data)
        output_hook(output.data)

    for name, m in model.named_modules():
        if tp.is_step_module(m):
            pos = name.find('.')
            if pos >= 0:
                name = name[pos+1:]
            tb_input_hook = tb_logger.register_histogram_hook("step_input/{}".format(name),
                                                              drop_percent=drop_percent, sample_percent=sample_percent)
            tb_output_hook = tb_logger.register_histogram_hook("step_output/{}".format(name),
                                                               drop_percent=drop_percent, sample_percent=sample_percent)
            m.register_forward_hook(partial(forward_tb_hook, input_hook=tb_input_hook, output_hook=tb_output_hook))


def to_np(x):
    return x.data.cpu().numpy()


if __name__ == '__main__':
    main()
