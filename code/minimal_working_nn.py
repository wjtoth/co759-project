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
        adversarial_eval_dataset = get_adversarial_dataset('cifar10', args)

    print('Creating network...')
    
    network = ConvNet4(nonlin=nn.ReLU)
    if args.cuda:
        network = network.cuda()
    
    criterion = losses.multiclass_hinge_loss
    optimizer = partial(optim.Adam, lr=0.001)
    #localsearch_optimizer = LocalSearchOptimizer(list(network.all_modules)[::-1],criterion)
    
    #target_optimizer = LocalSearchOptimizer
    target_optimizer = partial(GeneticOptimizer, num_candidates = 10, num_generations = 20)

    train(network, train_loader, criterion, optimizer, 
          args.epochs, target_optimizer=target_optimizer, use_gpu=args.cuda)
    print('Finished training')

    if args.adv_eval:
        print('Evaluating on adversarial examples...')
        failure_rate = evaluate_adversarially(
            network, adversarial_eval_dataset, 'untargeted_misclassify', 
            "fgsm", args.test_batch, num_classes)
        print('Failure rate: %.2f\%' % failure_rate)


class TargetPropOptimizer:
    
    def __init__(self, modules, sizes, loss_functions, state=[]):
        self.modules = modules
        self.sizes = sizes
        self.loss_functions = loss_functions
        self.state = list(state)

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
        self.evaluate_targets(train_step, module_index, 
                              input, label, target)
        return self.choose_targets(train_step, module_index, 
                                   input, label, target)

class LocalSearchOptimizer(TargetPropOptimizer):

    def __init__(self, modules, sizes, loss_functions, state=[]):
        TargetPropOptimizer.__init__(self, modules, sizes, loss_functions, state)
        self.number_candidates = 1

    def boxflip(self, candidate, x0, y0, x1, y1):   
        #for i in range(x0, x1):
        #    for j in range(y0, y1):
        #        candidate[i][j] = -1 * candidate[i][j]
        candidate[x0:x1,y0:y1] = -candidate[x0:x1,y0:y1]     
        return candidate

    def generate_candidate(self, module_index, target):
        #generate a target
        candidate_size = self.sizes[module_index]
                        
        candidate = torch.LongTensor()
        #print(candidate_size)
        candidate.resize_([64] + candidate_size)
        candidate.random_(0,2)
        candidate.add_(candidate)
        candidate.add_(-1)

        #Local Search Procedure
        best_flip = -1
        loss = self.evaluate_candidate(module_index, target, candidate)

        row_length = candidate.size()[1]
        search_steps = 10
        #walking to local optimum will take too long
        for k in range(0,search_steps):
            #flip Tensor in chunks of rows to reduce number of loss evaluations
            for i in range(0, 8):
                candidate = self.boxflip(candidate, 8*i, 0, 8*(i+1), row_length)
                new_loss = self.evaluate_candidate(module_index, target, candidate)
                candidate = self.boxflip(candidate, 8*i, 0, 8*(i+1), row_length)
                if new_loss.data[0] < loss.data[0]:
                    loss = new_loss
                    best_flip = i
            if best_flip > -1:
                candidate = self.boxflip(candidate, 8*best_flip, 0, 8*(best_flip+1), row_length)
        #set state to candidates
        self.state.append((Variable(candidate),loss))

    def generate_targets(self, train_step, module_index, input, label, target, base_targets=None):
        self.state=[]
        for i in range(0, self.number_candidates):    
            self.generate_candidate(module_index, target)

    
    def evaluate_candidate(self, module_index, target, candidate):
        loss_function = self.loss_functions[module_index]
        module = self.modules[module_index]
        output = module(Variable(candidate.type("torch.FloatTensor")))
        test_target = target
        loss_fn = loss_function
        if module_index > 0:
            loss_fn = torch.nn.MSELoss(size_average=False)
            test_target = test_target.type("torch.FloatTensor")
        return loss_fn(output, test_target)

    def evaluate_targets(self, train_step, module_index, input, label, target): 
        #NOOP Since generate targets does the work
        #candidate_loss_pairs = []

        #for candidate in self.state:
 #           loss = self.evaluate_candidate(module_index, target, candidate)
 #          candidate_loss_pairs.append((Variable(candidate),loss))
        #self.state = candidate_loss_pairs
        a=1
    
    def choose_targets(self, train_step, module_index, input, label, target):
        (best_candidate, best_loss) = self.state[0]
        for i in range(1, len(self.state)):
            (candidate, loss) = self.state[i]
            if loss.data[0] < best_loss.data[0]:
                best_candidate = candidate
                best_loss = loss
        return (best_candidate, best_loss)
    

class Target:
    def __init__(self, size, random=True):
        self.size = [64] + size
        self.values = torch.LongTensor()
        self.values.resize_(self.size)
        self.values.zero_()
        
        if random:
            self.values.random_(0,2)
            self.values.add_(self.values)
            self.values.add_(-1)

        self.loss = 0
        
    def evaluate(self, target_optimizer, module_index, labels):
        loss_function = target_optimizer.loss_functions[module_index]
        module = target_optimizer.modules[module_index]
        output = module(Variable(self.values.type("torch.FloatTensor")))
        test_target = labels
        loss_fn = loss_function
        if module_index > 0:
            loss_fn = torch.nn.MSELoss(size_average=False)
            test_target = test_target.type("torch.FloatTensor")
        self.loss = loss_fn(output, test_target)
        
    def crossover(self, parent_candidate1, parent_candidate2):
        x0 = randint(1,self.size[0]-2)
        y0 = randint(1,self.size[1]-2)
           
        self.values[:x0,:y0] = parent_candidate1.values[:x0,:y0] 
        self.values[x0+1:,:y0] = parent_candidate1.values[x0+1:,:y0] 

        self.values[:x0,y0+1:] = parent_candidate2.values[:x0,y0+1:] 
        self.values[x0+1:,y0+1:] = parent_candidate2.values[x0+1:,y0+1:] 
            
class GeneticOptimizer(TargetPropOptimizer):

    def __init__(self, modules, sizes, loss_functions, num_generations, num_candidates, state=[]):
        TargetPropOptimizer.__init__(self, modules, sizes, loss_functions, state)
        self.num_generations = num_generations
        self.num_candidates = num_candidates
    
    def boxflip(self, candidate, x0, y0, x1, y1):   
        candidate[x0:x1][y0:y1] = -candidate[x0:x1][y0:y1]     
        return candidate

    def generate_candidate(self, module_index, target):
        candidate_size = self.sizes[module_index]
        
        #generate a population of targets
        population = []
        for i in np.arange(0, self.num_candidates):
            #generate a random candidate target
            candidate = Target(candidate_size, random=True)
            candidate.evaluate(self, module_index, target)           
            population.append(candidate)
            
        #main loop
        for i in np.arange(0, self.num_generations):
            
            #generate num_candidates children
            for j in np.arange(0, self.num_candidates):
                #choose two random parents, and append crossover to population
                #some other condition for crossing two target setting could be used here
                c1 = randint(0, self.num_candidates-1)
                c2 = randint(0, self.num_candidates-1)
                while c1 == c2:
                    c1 = randint(0, self.num_candidates-1)
                    c2 = randint(0, self.num_candidates-1)
                
                child_candidate = Target(candidate_size, random=False)
                
                child_candidate.crossover(population[c1], population[c2])
                child_candidate.evaluate(self, module_index, target)            
                
                population.append(child_candidate)
                
            #mutate?
        
            #sort population based on loss
            population.sort(key=lambda target: float(target.loss), reverse = False)
            
            #kill kill kill
            population = population[0:self.num_candidates]
            
            #print("Generation: %d, best loss: %f"%(i,population[0].loss))
             
        #set state to candidates
        self.state.append((Variable(population[0].values),population[0].loss))

    def generate_targets(self, train_step, module_index, input, label, target, base_targets=None):
        self.state=[]
        for i in range(0, self.num_candidates):    
            self.generate_candidate(module_index, target)

    def evaluate_candidate(self, module_index, target, candidate):
        loss_function = self.loss_functions[module_index]
        module = self.modules[module_index]
        output = module(Variable(candidate.type("torch.FloatTensor")))
        test_target = target
        loss_fn = loss_function
        if module_index > 0:
            loss_fn = torch.nn.MSELoss(size_average=False)
            test_target = test_target.type("torch.FloatTensor")
        return loss_fn(output, test_target)

    def evaluate_targets(self, train_step, module_index, input, label, target): 
        #NOOP Since generate targets does the work
        #candidate_loss_pairs = []

        #for candidate in self.state:
 #           loss = self.evaluate_candidate(module_index, target, candidate)
 #          candidate_loss_pairs.append((Variable(candidate),loss))
        #self.state = candidate_loss_pairs
        a = 1
        
    def choose_targets(self, train_step, module_index, input, label, target):
        (best_candidate, best_loss) = self.state[0]
        for i in range(1, len(self.state)):
            (candidate, loss) = self.state[i]
            if loss.data[0] < best_loss.data[0]:
                best_candidate = candidate
                best_loss = loss
        return (best_candidate, best_loss)
        
def train(model, dataset_loader, loss_function, 
          optimizer, epochs, target_optimizer=None, use_gpu=True):
    train_per_layer = target_optimizer is not None
    if train_per_layer:
        optimizers = [optimizer(module.parameters()) for module in model.all_modules]
        modules = list(zip(model.all_modules, optimizers))[::-1]  # in reverse order
        target_optimizer = target_optimizer(
            modules=list(model.all_modules)[::-1],\
            sizes=model.input_sizes[::-1], \
            loss_functions=[loss_function]*len(model.all_modules))
    else:
        optimizer = optimizer(model.parameters())
    for epoch in range(epochs):
        last_time = time()
        for i, (inputs, labels, _) in enumerate(dataset_loader):
            if i % 10 == 1:
                if train_per_layer:
                    ouputs = model(Variable(inputs))
                    loss = loss_function(ouputs, Variable(labels))
                if i == 1:
                    steps = 1
                else:
                    steps = 10
                current_time = time() 
                print('epoch: %d, batch: %d, loss: %.3f, steps/sec: %.2f' 
                      % (epoch+1, i+1, loss.data[0], steps/(current_time-last_time)))
                last_time = current_time
            train_step = epoch*len(dataset_loader) + i
            inputs, labels = Variable(inputs), Variable(labels)
            
            if use_gpu:
                inputs, labels = inputs.cuda(), labels.cuda()
            targets = labels
            if train_per_layer:
                for j, (module, optimizer) in enumerate(modules):
                    optimizer.zero_grad()
                    if j == len(modules)-1:
                        # no target generation at initial layer/module
                        outputs = module(inputs)
                        loss = torch.nn.MSELoss(size_average=False)(outputs, targets.type("torch.FloatTensor"))
                        #loss = loss_function(outputs, targets)
                    else:
                        targets, loss = target_optimizer.step(
                            train_step, j, targets, label=labels)
                    loss.backward()
                    optimizer.step()
            else:
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = loss_function(outputs, targets)
                loss.backward()
                optimizer.step()

def get_adversarial_dataset(dataset_name, terminal_args):
    args = terminal_args
    _, adv_eval_loader, _, _ = create_datasets(
        dataset_name, args.batch, 1, not args.no_aug, 
        args.no_val, args.data_root, args.cuda, args.seed, 
        args.nworkers, args.dbg_ds_size, args.download)
    return [(image[0].numpy(), torch.LongTensor(label[0]).numpy()) 
            for image, label in adv_eval_loader]


def evaluate_adversarially(model, dataset, criterion, 
                           attack, batch_size, num_classes):
    model.eval()
    adversarial_examples = adversarial.generate_adversarial_examples(
        model, attack, dataset, criterion, 
        pixel_bounds=(-255, 255), num_classes=num_classes)
    failure_rate = adversarial.adversarial_eval(
        model, adversarial_examples, criterion, batch_size=batch_size)
    return failure_rate


class ConvNet4(nn.Module):
    def __init__(self, nonlin=nn.ReLU, use_batchnorm=False, input_shape=(3, 32, 32)):
        super(ConvNet4, self).__init__()
        self.use_batchnorm = use_batchnorm
        self.conv1_size = 32
        self.conv2_size = 64
        self.fc1_size = 1024
        self.fc2_size = 10

        self.input_sizes = [list(input_shape), [self.conv1_size,(input_shape[1] // 4)*(input_shape[2] // 4)//4, self.conv2_size//4], [(input_shape[1] // 4) * (input_shape[2] // 4) 
                              * self.conv2_size], [self.fc1_size]]

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
                grad_input = grad_input * go.size()[0] * go.abs()  
        return grad_input


class Step(nn.Module):
    def __init__(self, targetprop_rule=tp.TPRule.TruncWtHinge, 
                 make01=False, scale_by_grad_out=False):
        super(Step, self).__init__()
        self.tp_rule = targetprop_rule
        self.make01 = make01
        self.scale_by_grad_out = scale_by_grad_out
        self.step_function = StepF(targetprop_rule=self.tp_rule, make01=self.make01, 
                                   scale_by_grad_out=self.scale_by_grad_out)
        self.output_hook = None

    def __repr__(self):
        s = '{name}(a={a}, b={b}, tp={tp})'
        a = 0 if self.make01 else -1
        return s.format(name=self.__class__.__name__, a=a, b=1, tp=self.tp_rule)

    def register_output_hook(self, output_hook):
        self.output_hook = output_hook

    def forward(self, x):
        y = self.step_function(x)
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
