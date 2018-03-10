# from copy import deepcopy
from enum import Enum, unique
from functools import partial
import numpy as np

import torch


def sign11(x):
    """take the sign of the input, and set sign(0) = -1, so that output \in {-1, +1} always"""
    return torch.sign(x).clamp(min=0) * 2 - 1


def hinge(z, t, margin=1.0, trunc_thresh=float('inf'), scale=1.0):
    """compute hinge loss for each input (z) w.r.t. each target (t)"""
    loss = ((-z * t.float() + margin) * scale).clamp(min=0, max=trunc_thresh)  # .mean(dim=0).sum()
    return loss


def dhinge_dz(z, t, margin=1.0, trunc_thresh=float('inf'), norm_by_size=True):
    """compute derivative of hinge loss w.r.t. input z"""
    tz = z * t
    dhdz = (torch.gt(tz, margin - trunc_thresh) * torch.le(tz, margin)).float() * -t
    if norm_by_size:
        dhdz = dhdz * (1.0 / tz.size()[0])
    return dhdz


def hingeL2(z, t, margin=1.0, trunc_thresh=float('inf')):
    # loss = (-(z * t).float() + margin).clamp(min=0, max=trunc_thresh)
    loss = (margin - (z * t).float().clamp(min=trunc_thresh)).clamp(min=0)
    loss = loss * loss / 2
    return loss  # .mean(dim=0).sum()


def log_hinge(z, t, margin=1.0, trunc_thresh=float('inf'), scale=1.0):
    # loss = (torch.log(1.0 + margin - (z * t.float()).clamp(max=margin)) * scale).clamp(min=0, max=trunc_thresh)
    loss = (torch.log(1.0 + margin - (z * t.float()).clamp(min=-margin, max=margin)) * scale)
    return loss


def sigmoid(z, t, xscale=1.0, yscale=1.0):
    loss = torch.sigmoid(-(z * t).float() * xscale) * yscale
    return loss


def log_sigmoid(z, t, xscale=1.0, yscale=1.0):
    return torch.log(sigmoid(z, t, xscale, yscale))


def log_loss(z, t, trunc_thresh=float('inf')):
    loss = torch.log(1.0 + torch.exp(-z * t)).clamp(max=trunc_thresh)
    return loss


def square_loss(z, t, margin=1.0, scale=1.0, trunc_thresh=float('inf')):
    # loss = (min(z, 1)-1)^2
    loss = (((margin - (z * t).clamp(max=1)) ** 2) * scale).clamp(max=trunc_thresh)
    return loss


def soft_hinge(z, t, xscale=1.0, yscale=1.0):
    loss = yscale * torch.tanh(-(z * t).float() * xscale) + 1
    return loss


def hinge11(z, t, margin=1.0, trunc_thresh=2):
    loss = (-z * t.float() + margin).clamp(min=0, max=trunc_thresh) - 1.0
    return loss

# # @profile
# def dhinge_ab_dz(z, t, a=-1, b=1, step_thresh=0, margin=1, trunc_thresh=float('inf'), rescale_targets=False):
#     denom = 1. / (b - a)
#     z11 = (2 * denom) * (z - step_thresh) - 1
#     t11 = (2 * denom) * (t - a) - 1 if rescale_targets else t
#     tz11 = z11 * t11
#     # return (torch.ge(tz11, 0 - trunc_thresh) * torch.le(tz11, margin)).float() * -t11 * (1. / t.size()[0])
#     return (torch.ge(tz11, margin - trunc_thresh) * torch.le(tz11, margin)).float() * -t11 * (1. / t.size()[0])
#     # TODO: shouldn't this be tz11 > margin - trunc_thresh ??


def is_step_module(module):
    mstr = str(module)
    return mstr[0:5] == 'Step(' or mstr[0:10] == 'Staircase(' or mstr[0:13] == 'OldStaircase('


def needs_backward_twice(tp_rule):
    return (tp_rule == TPRule.SSTEAndWtHinge or
            tp_rule == TPRule.SSTEAndTruncWtHinge or
            tp_rule == TPRule.SSTEAndScaledTruncWtHinge)


@unique
class TPRule(Enum):
    # the different targetprop rules for estimating targets and updating weights
    WtHinge = 0
    WtL2Hinge = 1
    WeightedPerceptron = 2
    Adaline = 3
    STE = 4
    SSTE = 5
    SSTEAndWtHinge = 6
    SaturatingWtHinge = 7
    TruncWtHinge = 8
    SSTEAndTruncWtHinge = 9
    SSTEAndScaledTruncWtHinge = 10
    Trunc2WtHinge = 11
    TruncWtHingeSSTE = 12
    TruncWtHingeHeur = 13
    # GreedyTruncWtHinge = 14
    LogWtHinge = 14
    TruncWtL2Hinge = 15
    TruncLogWtHinge = 16
    TruncWtPerceptron = 17
    Sigmoid = 18
    LogLoss = 19
    TruncLogLoss = 20
    SquareLoss = 21
    TruncSquareLoss = 22
    TruncWtHinge2 = 23
    Sigmoid2_2 = 24
    LogSigmoid = 25
    TruncWtHinge11 = 26
    SoftHinge = 27
    SoftHinge2 = 28
    Sigmoid3_2 = 29
    SSTEv2 = 30
    LeCunTanh = 31
    Ramp = 32

    @staticmethod
    def wt_hinge_backward(step_input, grad_output, target, is01):
        # if target is not None:
        #    print('num diff targets:', torch.sum(torch.ne(-torch.sign(grad_output), target)))
        if target is None:
            target = -torch.sign(grad_output)
            # #target[torch.eq(grad_output, 0)] = torch.sign(step_input) # TODO: remove -- enforces no zero targets
        # #return torch.mul(torch.lt(step_input * target, 1).float(), -target) / float(step_input.size()[0]), target
        # # return dhinge_dz(step_input*target, target, m=1), target
        # # return torch.abs(grad_output) * torch.mul(torch.lt(step_input * target, 1).float(), -target), target
        # return dhinge_ab_dz(step_input, target, a=0 if is01 else -1), None
        assert False
        return dhinge_dz(step_input, target, margin=1), None

    @staticmethod
    def wt_l2_hinge_backward(step_input, grad_output, target, is01):
        assert False
        assert not is01, 'wt_l2_hinge_backward doesn''t support is01 yet'
        target = -torch.sign(grad_output) if target is None else target
        return (-step_input * target + 1.0).clamp(min=0) * -target * (1.0 / step_input.size()[0]), None

    @staticmethod
    def wt_perceptron_backward(step_input, grad_output, target, is01):
        assert not is01
        target = -torch.sign(grad_output) if target is None else target
        # #return torch.mul(torch.lt(step_input * target, 0).float(), -target) / float(step_input.size()[0]), target
        # # return dhinge_dz(step_input*target, target, m=0), target
        # return dhinge_ab_dz(step_input, target, a=(0 if is01 else -1), margin=0), None
        return dhinge_dz(step_input, target, margin=0), None

    @staticmethod
    def adaline_backward(step_input, grad_output, target, is01):
        assert not is01, 'adaline backward doesn''t support is01 yet'
        target = -torch.sign(grad_output) if target is None else target
        return torch.mul(torch.abs(target), step_input - target) * (1.0 / step_input.size()[0]), None

    @staticmethod
    def ste_backward(step_input, grad_output, target, is01):
        return grad_output, None

    @staticmethod
    def sste_backward(step_input, grad_output, target, is01, a=1):
        # is01 = True
        if is01:
            grad_input = grad_output * torch.ge(step_input, 0).float() * torch.le(step_input, a).float()
        else:
            grad_input = grad_output * torch.le(torch.abs(step_input), a).float()
        # grad_input /= 2
        return grad_input, None

    @staticmethod
    def saturating_wt_hinge_backward(step_input, grad_output, target, is01):
        assert False, 'use truncated wt hinge backward'
        grad_in, target = TPRule.wt_hinge_backward(step_input, grad_output, target, is01)
        grad_in *= torch.le(torch.abs(step_input), 1).float()
        return grad_in, target

    # @profile
    @staticmethod
    def trunc_wt_hinge_backward(step_input, grad_output, target, is01):
        if target is None:
            target = -torch.sign(grad_output)
        assert not is01, 'is01 not supported'
        grad_input = dhinge_dz(step_input, target, margin=1, trunc_thresh=2)
        return grad_input, None

    @staticmethod
    def trunc2_wt_hinge_backward(step_input, grad_output, target, is01):
        assert False
        if target is None:
            target = -torch.sign(grad_output)
            # target[torch.eq(grad_output, 0)] = torch.sign(step_input) # TODO: remove -- enforces no zero targets
        assert not is01, 'is01 not supported'
        grad_input = dhinge_dz(step_input, target, margin=1, trunc_thresh=3)
        return grad_input, None

    @staticmethod
    def wt_hinge_and_sste_backward(step_input, grad_output, target, is01):
        assert False
        if target is None:
            grad_input, _ = TPRule.sste_backward(step_input, grad_output, target, is01)
            target = -torch.sign(grad_output)
        else:
            grad_input, target = TPRule.wt_hinge_backward(step_input, grad_output, target, is01)
            # grad_input *= torch.abs(grad_output) * float(step_input.size()[0]) # TODO: REMOVE THIS
            # grad_input = grad_output*torch.le(torch.abs(step_input), 1).float() # TODO: REMOVE THIS
        return grad_input, target

    @staticmethod
    def trunc_wt_hinge_and_sste_backward(step_input, grad_output, target, is01):
        assert False
        if target is None:
            grad_input, _ = TPRule.sste_backward(step_input, grad_output, target, is01)
            target = -torch.sign(grad_output)
        else:
            grad_input, target = TPRule.trunc_wt_hinge_backward(step_input, grad_output, target, is01)
        return grad_input, target

    @staticmethod
    def scaled_trunc_wt_hinge_and_sste_backward(step_input, grad_output, target, is01):
        assert False
        if target is None:
            grad_input, _ = TPRule.sste_backward(step_input, grad_output, target, is01)
            target = grad_output  # save the scale and direction of the incoming gradient as the target
        else:
            # use the TWH target but then scale the backprop gradient based on the stored gradient
            grad_input, _ = TPRule.trunc_wt_hinge_backward(step_input, grad_output, -torch.sign(grad_output), is01)
            grad_input = grad_input * torch.abs(target)
        return grad_input, target

    @staticmethod
    def trunc_wt_hinge_sste_backward(step_input, grad_output, target, is01):
        assert False
        assert not is01
        if target is None:
            target = -grad_output
        grad_input = dhinge_dz(step_input, target, margin=1, trunc_thresh=2)
        return grad_input, None

    @staticmethod
    def trunc_wt_hinge_heur_backward(step_input, grad_output, target, is01):
        assert False, 'this is wrong; need to determine current targets differently; this just assumes the forward ' \
                      'pass chose correctly, which is not true'
        assert not is01
        if target is None:
            z, t_cur = step_input, torch.sign(step_input)
            # compute gradient w.r.t. target (dh/dz is symmetric w.r.t. t and z, so it's ok)
            upstream_contrib = dhinge_dz(t_cur, z, margin=1, trunc_thresh=2, norm_by_size=False)
            downstream_contrib = -grad_output #* z.size(0)
            target = torch.sign(upstream_contrib * 1e-6 + downstream_contrib)
        grad_input = dhinge_dz(step_input, target, margin=1, trunc_thresh=2)
        return grad_input, None

    @staticmethod
    def sigmoid_backward(step_input, grad_output, target, is01, xscale=2.0, yscale=1.0):
        assert not is01
        if target is None:
            target = torch.sign(-grad_output)
        z = sigmoid(step_input, target, xscale=xscale, yscale=1.0)
        grad_input = z * (1 - z) * xscale * yscale * -target / grad_output.size(0)
        return grad_input, None

    @staticmethod
    def tanh_backward(step_input, grad_output, target, is01, xscale=1.0, yscale=1.0):
        # assert not is01
        if target is None:
            target = torch.sign(-grad_output)
        z = soft_hinge(step_input, target, xscale=xscale, yscale=1.0) - 1
        grad_input = (1 - z * z) * xscale * yscale * -target / grad_output.size(0)
        return grad_input, None

    @staticmethod
    def ramp_backward(step_input, grad_output, target, is01):
        if target is None:
            target = torch.sign(-grad_output)
        abs_input = torch.abs(step_input)
        if is01:
            # grad_input = grad_output * ((step_input <= 1).float() * (step_input >= 0).float() +
            #                             abs_input * (step_input > -1).float() * (step_input < 0).float() +
            #                             (2 - abs_input) * (step_input < 2).float() * (step_input > 1).float())
            # ramp01 = @(zt) (0 <= zt) .* (zt <= 1) + ...
            #                 (zt + 1) .* (-1 < zt) .* (zt < 0) + ...
            #                 (2 - abs(zt)) .* (1 < zt) .* (zt < 2);
            ramp_input = ((0 <= step_input).float() * (step_input <= 1).float() +
                          (step_input+1) * (-1 < step_input).float() * (step_input < 0).float() +
                          (2 - abs_input) * (1 < step_input).float() * (step_input < 2).float())
        else:
            # grad_input = grad_output * ((abs_input <= 1).float() +
            #                             (2 - abs_input) * (abs_input < 2).float() * (abs_input > 1).float())
            # ramp = @(zt) ((abs(zt) <= 1) + ...
            #                 abs(2 - zt) .* (zt < 2) .* (zt > 1) + ...
            #                 abs(zt + 2) .* (zt < -1) .* (zt > -2));
            ramp_input = ((abs_input <= 1).float() +
                          (2 - step_input).abs_() * (1 < step_input).float() * (step_input < 2).float() +
                          (2 + step_input).abs_() * (-2 < step_input).float() * (step_input < -1).float())
        grad_input = grad_output * ramp_input
        return grad_input, None


    @staticmethod
    def get_backward_func(targetprop_rule):
        if targetprop_rule == TPRule.WtHinge:  # gradient of hinge loss
            tp_grad_func = TPRule.wt_hinge_backward
        elif targetprop_rule == TPRule.WtL2Hinge:  # gradient of squared hinge loss
            tp_grad_func = TPRule.wt_l2_hinge_backward
        elif targetprop_rule == TPRule.WeightedPerceptron:  # gradient of perceptron criterion
            tp_grad_func = TPRule.wt_perceptron_backward
        elif targetprop_rule == TPRule.Adaline:  # adaline / delta-rule update
            tp_grad_func = TPRule.adaline_backward
        elif targetprop_rule == TPRule.STE:
            tp_grad_func = TPRule.ste_backward
        elif targetprop_rule == TPRule.SSTE:
            tp_grad_func = TPRule.sste_backward
        elif targetprop_rule == TPRule.SSTEAndWtHinge:     # use SSTE to estimate targets and then do weight
                                                        # updates based on per-layer weighted hinge loss
            tp_grad_func = TPRule.wt_hinge_and_sste_backward
        elif targetprop_rule == TPRule.SaturatingWtHinge:
            tp_grad_func = TPRule.saturating_wt_hinge_backward
        elif targetprop_rule == TPRule.TruncWtHinge:  # or targetprop_rule == TPRule.GreedyTruncWtHinge:
                                                            # weighted hinge using a truncated hinge loss
            tp_grad_func = TPRule.trunc_wt_hinge_backward
        elif targetprop_rule == TPRule.SSTEAndTruncWtHinge:
            tp_grad_func = TPRule.trunc_wt_hinge_and_sste_backward
        elif targetprop_rule == TPRule.SSTEAndScaledTruncWtHinge:  # use TWH to estimate targets and then scale
                                                                # the weight updates based on abs(SSTE)
            tp_grad_func = TPRule.scaled_trunc_wt_hinge_and_sste_backward
        elif targetprop_rule == TPRule.Trunc2WtHinge:
            tp_grad_func = TPRule.trunc2_wt_hinge_backward
        elif targetprop_rule == TPRule.TruncWtHingeSSTE:
            tp_grad_func = TPRule.trunc_wt_hinge_sste_backward
        elif targetprop_rule == TPRule.TruncWtHingeHeur:
            tp_grad_func = TPRule.trunc_wt_hinge_heur_backward
        elif targetprop_rule == TPRule.Sigmoid:
            tp_grad_func = TPRule.sigmoid_backward
        elif targetprop_rule == TPRule.Sigmoid3_2:
            tp_grad_func = partial(TPRule.sigmoid_backward, xscale=3.0, yscale=2.0)
        elif targetprop_rule == TPRule.Sigmoid2_2:
            tp_grad_func = partial(TPRule.sigmoid_backward, xscale=2.0, yscale=2.0)
        elif targetprop_rule == TPRule.SoftHinge:
            tp_grad_func = partial(TPRule.tanh_backward, xscale=1.0)
        elif targetprop_rule == TPRule.LeCunTanh:
            tp_grad_func = partial(TPRule.tanh_backward, xscale=(2.0/3.0), yscale=1.7519)
        elif targetprop_rule == TPRule.SSTEv2:
            a = np.sqrt(12.0) / 2
            tp_grad_func = partial(TPRule.sste_backward, a=a)
        elif targetprop_rule == TPRule.Ramp:
            tp_grad_func = TPRule.ramp_backward
        else:
            raise ValueError('specified targetprop rule ({}) has no backward function'.format(targetprop_rule))
        return tp_grad_func

    @staticmethod
    def get_loss_func(targetprop_rule):
        if targetprop_rule == TPRule.WtHinge:
            tp_loss_func = hinge
        elif targetprop_rule == TPRule.TruncWtHinge:  # or targetprop_rule == TPRule.GreedyTruncWtHinge:
            tp_loss_func = partial(hinge, trunc_thresh=2)
        elif targetprop_rule == TPRule.TruncWtHinge2:
            # tp_loss_func = partial(hinge, trunc_thresh=2, scale=0.5)
            tp_loss_func = partial(hinge, trunc_thresh=2, scale=2)
        elif targetprop_rule == TPRule.Trunc2WtHinge:
            # tp_loss_func = partial(hinge, trunc_thresh=3)
            # tp_loss_func = partial(hinge, trunc_thresh=1)
            # tp_loss_func = partial(hinge, margin=2, trunc_thresh=4)
            tp_loss_func = partial(hinge, margin=0.5, trunc_thresh=1)
        elif targetprop_rule == TPRule.TruncWtHinge11:
            tp_loss_func = hinge11
        elif targetprop_rule == TPRule.WtL2Hinge:
            tp_loss_func = hingeL2
        elif targetprop_rule == TPRule.TruncWtL2Hinge:
            tp_loss_func = partial(hingeL2, trunc_thresh=-1)
        elif targetprop_rule == TPRule.LogWtHinge:
            tp_loss_func = log_hinge
        elif targetprop_rule == TPRule.TruncLogWtHinge:
            # tp_loss_func = partial(log_hinge, trunc_thresh=2)
            # tp_loss_func = partial(log_hinge, trunc_thresh=2, scale=0.5)
            tp_loss_func = partial(log_hinge, trunc_thresh=2, scale=2)
        elif targetprop_rule == TPRule.TruncWtPerceptron:
            tp_loss_func = partial(hinge, margin=0, trunc_thresh=1)
        elif targetprop_rule == TPRule.Sigmoid:
            tp_loss_func = partial(sigmoid, xscale=2.0)
            # tp_loss_func = partial(sigmoid, xscale=1.0)
        elif targetprop_rule == TPRule.Sigmoid2_2:
            tp_loss_func = partial(sigmoid, xscale=2.0, yscale=2.0)
        elif targetprop_rule == TPRule.Sigmoid3_2:
            tp_loss_func = partial(sigmoid, xscale=3.0, yscale=2.0)
        elif targetprop_rule == TPRule.LogSigmoid:
            tp_loss_func = partial(log_sigmoid, xscale=2.0, yscale=1.0)
        elif targetprop_rule == TPRule.LogLoss:
            tp_loss_func = log_loss
        elif targetprop_rule == TPRule.TruncLogLoss:
            tp_loss_func = partial(log_loss, trunc_thresh=2)
        elif targetprop_rule == TPRule.SquareLoss:
            tp_loss_func = square_loss
        elif targetprop_rule == TPRule.TruncSquareLoss:
            tp_loss_func = partial(square_loss, trunc_thresh=4)
        elif targetprop_rule == TPRule.SoftHinge:
            tp_loss_func = soft_hinge
        elif targetprop_rule == TPRule.SoftHinge2:
            tp_loss_func = partial(soft_hinge, xscale=2.0)
        elif targetprop_rule == TPRule.LeCunTanh:
            tp_loss_func = partial(soft_hinge, xscale=(2.0 / 3.0), yscale=1.7519)
        else:
            raise ValueError('targetprop rule ({}) does not have an associated loss function'.format(targetprop_rule))
        return tp_loss_func
