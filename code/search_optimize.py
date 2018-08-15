import numpy as np
from functools import partial

import torch


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


def infer(modules, input_, activations=None):
    if isinstance(modules, list):
        for i, module in enumerate(reversed(list(modules))):
            if i == 0:
                output = module(input_)
            else:
                output = module(activations[i-1](output))
    else:
        output = modules(input_)
    return output


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

    def evaluate_candidates(self, modules, module_index, loss_function, targets, 
                            candidates, activations=None, criterion_weight=None):
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
        output = infer(modules, candidate_batch, activations)
        if module_index > 0 and self.criterion != "output_loss":
            output = output.view_as(target_batch)
            if self.criterion.startswith("accuracy"):
                output = self.activations[module_index](output)
            if self.criterion == "accuracy_top5":
                raise RuntimeError("Can only use top 5 accuracy as "
                                   "optimization criteria at output layer.")
        if "loss" in self.criterion:
            losses = loss_function(output, target_batch)
        elif self.criterion == "accuracy":
            losses = accuracy(output, target_batch, average=False)
        elif self.criterion == "accuracy_top5":
            losses = accuracy_topk(output, target_batch, average=False)
        if criterion_weight is not None:
            criterion_weight = criterion_weight.view(
                criterion_weight.shape[0]*criterion_weight.shape[1],
                *criterion_weight.shape[2:])
            losses *= criterion_weight
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

    def step(self, train_step, module_index, module_target, 
             base_targets=None, criterion_weight=None):
        return self.generate_target(train_step, module_index, module_target, 
                                    base_targets, criterion_weight)


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
    batch_base = torch.stack([base_tensor]*size)
    sampling_prob = radius / base_tensor.numel()
    if perturb_base_tensor is None:
        one_tensor = torch.ones(batch_base.shape)
        if use_gpu:
            one_tensor = one_tensor.cuda()
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
    candidate = torch.randint(0, 2, shape)
    if use_gpu:
        candidate = candidate.cuda()
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


def inverse_receptive_fields(input, output, field_count, layer, model_info):
    # Assumes first batch dimension and splits along third and fourth dimensions
    conv_kernel = model_info[layer]["conv"]["kernel_size"]
    padding = model_info[layer]["conv"]["padding"]
    pool_kernel = model_info[layer]["max_pool"]["kernel_size"]
    pass


def normalized_diff(x, y, offset=3/5):
    sign_match = torch.eq(x.sign(), y.sign()).float()
    shifted_norm = torch.abs(x) + offset
    return sign_match * torch.clamp(shifted_norm, 0, 1)


def sample_sign(x, y):
    sampling_weights = normalized_diff(x, y, offset=5/5)
    sample_base = (torch.bernoulli(sampling_weights)*2) - 1
    return y * (-sample_base)


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
            torch.ones([self.candidates] + shape).cuda() if self.use_gpu
            else torch.ones([self.candidates] + shape) for shape in self.shapes]


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
                    sampling_weights=sample_tensor,
                    use_gpu=self.use_gpu))
        return torch.cat(candidates)

    def find_candidates(self, module_index, module_target, train_step, 
                        base_candidates=None, criterion_weight=None):
        loss_function = self.loss_functions[module_index]
        if self.criterion == "output_loss":
            modules = self.modules[:module_index+1]
            activations = self.activations[:module_index]
        else:
            modules = self.modules[module_index]
            activations = None
        if criterion_weight is not None:
            criterion_weight = torch.stack([criterion_weight]*(self.candidates 
                + (len(base_candidates) if base_candidates is not None else 1))) 
        if module_index > 0 and self.criterion != "output_loss":
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
            perturb_base_tensors = [
                torch.ones([count] + self.shapes[module_index]).cuda() if self.use_gpu
                else torch.ones([count] + self.shapes[module_index])
                for count, _ in search_parameters]
        else:
            perturb_base_tensors = [self.perturb_base_tensors[module_index]]

        # Search to find good candidate
        sampling_weights = None
        best_candidates = [(parent, None) for parent in parents]
        for i in range(self.iterations):
            iteration_parameters = [
                (count, perturb_scheduler(train_step, module_index, i))
                for count, perturb_scheduler in search_parameters]
            if self.candidate_grad_delay == i and self.candidate_type.startswith("grad"):
                if i == 0:
                    # Assumes first parent is activation
                    loss_grad = [self.evaluate_candidates(
                        modules, module_index, loss_function, 
                        module_target.unsqueeze(0), parents[0].unsqueeze(0), 
                        activations=activations)[1]]
                else:
                    loss_grad = group_data[2]
                candidate_batch = torch.cat(loss_grad, dim=0)
                if self.candidate_type == "grad_sampled":
                    # Assumes first parent is activation 
                    candidate_batch = sample_sign(
                        candidate_batch, parents[0].unsqueeze(0))
                else:
                    candidate_batch = -candidate_batch
                parents = ([torch.sign(torch.mean(candidate_batch, 0))] 
                           + (parents[1:] if i == 0 else []))
                best_candidates = [(parent, None) for parent in parents]
                continue
            else:
                candidate_batch = self.generate_candidates(
                    parents, iteration_parameters, 
                    perturb_base_tensors, sampling_weights)
            loss_data = self.evaluate_candidates(
                modules, module_index, loss_function, target_batch, 
                candidate_batch, activations, criterion_weight)
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

    def generate_target(self, train_step, module_index, module_target, 
                        base_targets=None, criterion_weight=None):
        self.state["candidates"] = []
        for i in range(self.searches):
            self.find_candidates(module_index, module_target, train_step, 
                                 base_targets, criterion_weight)
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

    def generate_target(self, train_step, module_index, module_target, 
                        base_targets=None, criterion_weight=None):
        self.state["candidates"] = []
        for i in range(self.searches):
            self.find_candidates(module_index, module_target, 
                                 train_step, base_targets)
        return self.state["candidates"][0], None