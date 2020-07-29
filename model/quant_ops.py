import collections
import math
import pdb
import random
import time
from itertools import repeat

import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Function as F

def _ntuple(n):
    def parse(x):
        if isinstance(x, collections.Iterable):
            return x
        return tuple(repeat(x, n))
    return parse

_pair = _ntuple(2)

def TorchRound():
  class identity_grad(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        return torch.round(input)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output

  return identity_grad().apply

def quant_max(tensor):
    """
    Returns the max value for symmetric quantization.
    """
    return torch.abs(tensor.detach()).max() + 1e-8

class quant_weight(nn.Module):
    """
    Quantization function for quantize weight with maximum.
    """
    def __init__(self, k_bits):
        super(quant_weight, self).__init__()
        self.k_bits = k_bits
        self.qmax = 2. ** (self.k_bits -1) - 1.
        self.round = TorchRound()

    def forward(self, input):
        max_val  = quant_max(input)
        q_weight = self.round(input * self.qmax / max_val) 
        q_weight = q_weight * max_val / qmax 
        return q_weight

class pams_quant_act(nn.Module):
    """
    Quantization function for quantize activation with parameterized max scale.
    """
    def __init__(self, k_bits, ema_epoch=1):
        super(pams_quant_act, self).__init__()
        self.deacy = 0.9997
        self.k_bits = k_bits
        self.qmax = 2. ** (self.k_bits -1) -1.
        self.mode = mode
        self.round = TorchRound()
        self.alpha = nn.Parameter(torch.Tensor(1))
        self.epoch = 1
        self.ema_epoch = ema_epoch
        self.register_buffer('max_val', torch.ones(1))

    def reset_parameter(self):
        nn.init.constant_(self.alpha, 10)

    def forward(self, x):
        if self.epoch > self.ema_epoch or not self.training:
            act = torch.max(torch.min(x, self.alpha), -self.alpha)
        
        elif self.epoch <= self.ema_epoch and self.training:
            self._ema(x)
            self.alpha.data = self.max_val.unsqueeze(0)
        
        act = act * self.qmax / self.alpha
        q_act = self.round(act)
        q_act = q_act * self.alpha / self.qmax
    
        return q_act

    def _ema(self, x):
        max_internel = torch.mean(torch.max(torch.max(torch.max(abs(x),dim=1)[0],dim=1)[0],dim=1)[0])
        max_val = torch.abs(tensor.detach()).max()
        if self.epoch == 0:
            self.max_val = max_val
        else:
            self.max_val = (1.0-self.decay) * max_val + decay * self.max_val

class QuantConv2d(nn.Module):
    """
    A convolution layer with quantized weight.
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=False, k_bits=32):
        super(QuantConv2d, self).__init__()
        self.weight = nn.Parameter(torch.Tensor(out_channels,in_channels,kernel_size,kernel_size))
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.in_channels = in_channels
        self.kernel_size = _pair(kernel_size)
        self.bias = None
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_channels))

        self.k_bits = k_bits
        self.quant_weight = quant_weight(k_bits = k_bits)
        self.output = None
        self.reset_parameters()

    def reset_parameter(self):
        stdv = 1.0/ math.sqrt(self.weight.size(0))
        self.weight.data.uniform_(-stdv,stdv)
        if self.bias is not None:
            nn.init.constant_(self.bias,0.0)

    def forward(self, input, order=None):
        return nn.functional.conv2d(input, self.quant_weight(self.weight), self.bias, self.stride, self.padding, self.dilation, self.groups)

def conv3x3(in_channels, out_channels,kernel_size=3,stride=1,padding=1,bias=False):
    return nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=padding, bias=bias)


def quant_conv3x3(in_channels, out_channels, kernel_size=3,padding = 1, stride=1, k_bits=32, bias = False):
    return QuantConv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride = stride, padding=padding, k_bits=k_bits, bias=bias)
