#!/usr/bin/python3.6  
# -*- coding: utf-8 -*-

from torch.autograd import Function as F
import torch.nn as nn
import torch
import math
import numpy as np
import pdb
import collections
from itertools import repeat
import time
import random

RoundGradient_LIST = ['linear','max','tanh','log2','gaussian','google']

global TRAIN_LAYER_NUM
TRAIN_LAYER_NUM = 0
global TEST_LAYER_NUM
TEST_LAYER_NUM = 0
global ACTIVATION_MIN 
FLAG = 1e5
FLAGLEN = 200
ACTIVATION_MIN = [FLAG for i in range(FLAGLEN)]
ACTIVATION_MAX = [FLAG for i in range(FLAGLEN)]
EMA_EPOCH = 1

def _ntuple(n):
    def parse(x):
        if isinstance(x, collections.Iterable):
            return x
        return tuple(repeat(x, n))
    return parse

_pair = _ntuple(2)

def quantize_max(tensor, num_bits):
    qmax = 2. ** (num_bits -1) -1.
    tensor = abs(tensor.detach())
    max_val = torch.where(tensor.max()>0,tensor.max(),torch.zeros_like(tensor.max()))+1e-8
    return max_val, qmax

def get_extremum(tensor, training, deacy, version, name):
    tensor = tensor.detach()
    # print(tensor)

    if name == 'vgg16':
        length = 14
    elif name == 'resnet56':
        length = 54

    elif name == 'edsr':
        length = 48

    elif name == 'rdn':
        length = 160
    else:
        length = 200

    if version == 1:
        min_value = torch.min(tensor)
        max_value = torch.max(tensor)

        return min_value, max_value

    elif version == 2:
        if len(tensor.shape) == 2:
            max_internel = torch.mean(torch.max(tensor,dim=1)[0])
            min_internel = torch.mean(torch.min(tensor,dim=1)[0])
        else:
            max_internel = torch.mean(torch.max(torch.max(torch.max(tensor,dim=1)[0],dim=1)[0],dim=1)[0])
            min_internel = torch.mean(torch.min(torch.min(torch.min(tensor,dim=1)[0],dim=1)[0],dim=1)[0])
        min_value = min_internel
        max_value = max_internel

    elif version == 3:
        if len(tensor.shape) == 2:
            max_internel = torch.mean(torch.max(abs(tensor),dim=1)[0])
            min_internel = torch.mean(torch.min(abs(tensor),dim=1)[0])
        else:
            max_internel = torch.mean(torch.max(torch.max(torch.max(abs(tensor),dim=1)[0],dim=1)[0],dim=1)[0])
            min_internel = torch.mean(torch.min(torch.min(torch.min(abs(tensor),dim=1)[0],dim=1)[0],dim=1)[0])
        min_value, max_value =   torch.where((min_internel)<0.,min_internel,\
                                torch.zeros_like(min_internel)),\
                                torch.where(max_internel>0.,max_internel,torch.zeros_like(max_internel))

    global TRAIN_LAYER_NUM
    global TEST_LAYER_NUM
    global ACTIVATION_MAX
    global ACTIVATION_MIN

    if training:
        TRAIN_LAYER_NUM += 1
        if TRAIN_LAYER_NUM > length: 
            TRAIN_LAYER_NUM = 1

        if ACTIVATION_MAX[TRAIN_LAYER_NUM-1] == FLAG:
            ACTIVATION_MAX[TRAIN_LAYER_NUM-1] = max_value
        else:
            ACTIVATION_MAX[TRAIN_LAYER_NUM-1] = (1.0-deacy) * max_value + deacy * ACTIVATION_MAX[TRAIN_LAYER_NUM-1]
        
        if ACTIVATION_MIN[TRAIN_LAYER_NUM-1] == FLAG:
            ACTIVATION_MIN[TRAIN_LAYER_NUM-1] = min_value
        else:
            ACTIVATION_MIN[TRAIN_LAYER_NUM-1] = (1.0-deacy) * min_value + deacy * ACTIVATION_MIN[TRAIN_LAYER_NUM-1]
        min_value = ACTIVATION_MIN[TRAIN_LAYER_NUM-1]
        max_value = ACTIVATION_MAX[TRAIN_LAYER_NUM-1]
        return min_value.cuda(), max_value.cuda()

    else:
        TEST_LAYER_NUM += 1
        if TEST_LAYER_NUM > length:
            TEST_LAYER_NUM = 1
        min_value = ACTIVATION_MIN[TEST_LAYER_NUM-1]
        max_value = ACTIVATION_MAX[TEST_LAYER_NUM-1]
        return min_value, max_value

def identity_gradient_function(k, mode, **kargs):
  class identity_quant_function(torch.autograd.Function):
    global RoundGradient_LIST
    @staticmethod
    def forward(ctx, input):
        if mode == 'dorefa':
            n = float(2 ** k - 1)
            out = torch.round(input * n) / n
        elif mode in RoundGradient_LIST:
            out = torch.round(input)
        else:
            print(f"The Mode of Quantification { mode } is not yet implemented")
        return out

    @staticmethod
    def backward(ctx, grad_output):
    #   grad_input = grad_output.clone()
      return grad_output

  return identity_quant_function().apply

class weight_quantize(nn.Module):
  def __init__(self, k_bits, mode):
    super(weight_quantize, self).__init__()
    self.k_bits = k_bits
    self.max_val = nn.Parameter(torch.Tensor(1), requires_grad=False)
    self.min_val = nn.Parameter(torch.Tensor(1), requires_grad=False)
    self.s = nn.Parameter(torch.Tensor(1), requires_grad=False)
    self.z = nn.Parameter(torch.Tensor(1), requires_grad=False)
    self.mode = mode
    self.uniform_q = identity_gradient_function(k=k_bits, mode=mode)

  def forward(self, input):
    max_val, qmax  = quantize_max(input,self.k_bits)
    weight = input * qmax / max_val
    weight_q = self.uniform_q(weight)
    weight_q = weight_q * max_val / qmax
    return weight_q

class activation_quantize(nn.Module):
  def __init__(self, k_bits,mode='linear',version=1,name='resnet'):
    super(activation_quantize, self).__init__()
    # assert k_bits % 2 ==0
    self.deacy = 0.95
    self.k_bits = k_bits
    self.version = version
    self.mode = mode
    self.uniform_q = identity_gradient_function(k=self.k_bits,mode=mode)
    self.activation = None
    self.qmin = nn.Parameter(torch.Tensor(1), requires_grad=False)
    self.qmax = nn.Parameter(torch.Tensor(1), requires_grad=False)
    self.s = nn.Parameter(torch.Tensor(1), requires_grad=False)
    self.z = nn.Parameter(torch.Tensor(1), requires_grad=False)

    self.name = name

  def reset_parameter(self):
    nn.init.constant_(self.alpha,1.0)

  def forward(self, input):
    self.qmax = 2. ** (self.k_bits -1) -1.
    max_val, self.qmax = quantize_max(input, self.k_bits)
    activation = input * self.qmax / max_val
    activation_q = self.uniform_q(activation)
    ctivation_q = activation_q * max_val /self.qmax

    return activation_q

class pact_activation_quantize(nn.Module):
  def __init__(self, k_bits,mode='linear',version=1,name='resnet', ema_epoch=1):
    super(pact_activation_quantize, self).__init__()
    # assert k_bits % 2 ==0
    self.deacy = 0.9997
    self.k_bits = k_bits
    self.qmax = 2. ** (self.k_bits -1) -1.
    # print('pact act bit', self.k_bits)
    self.version = version
    self.mode = mode
    self.uniform_q = identity_gradient_function(k=k_bits,mode=mode)
    self.name = name
    self.alpha = nn.Parameter(torch.Tensor(1))
    self.reset_parameter()
    self.K = 1e+5
    self.epoch = 1
    self.ema_epoch = ema_epoch

  def reset_parameter(self):
    nn.init.constant_(self.alpha, 10)

  def forward(self, x):
    if self.epoch > self.ema_epoch or not self.training:
        activation = torch.max(torch.min(x, self.alpha), -self.alpha)
    
    elif self.epoch <= self.ema_epoch and self.training:
        activation = x
        self.qmin, max_val = get_extremum(activation, self.training, self.deacy, self.version,self.name)
        self.alpha.data = max_val.unsqueeze(0)
    
    activation = activation * self.qmax / self.alpha
    activation_q = self.uniform_q(activation)
    activation_q = activation_q * self.alpha / self.qmax
   
    return activation_q

class QuantConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                    padding=0, dilation=1, groups=1, bias=False,k_bits=32,mode='linear'):
        super(QuantConv2d, self).__init__()
        self.weight = nn.Parameter(torch.Tensor(out_channels,in_channels,kernel_size,kernel_size))
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.in_channels = in_channels
        self.kernel_size = _pair(kernel_size)
        self.bias_flag = bias
        if self.bias_flag:
            self.bias = nn.Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias',None)
        self.k_bits = k_bits
        self.quantize_weight = weight_quantize(k_bits = k_bits,mode=mode)
        self.output = None
        self.reset_parameters()

    def reset_parameters(self):
        n = self.in_channels
        for k in self.kernel_size:
            n *= k
        stdv = 1. / math.sqrt(n)
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def reset_parameter(self):
        stdv = 1.0/ math.sqrt(self.weight.size(0))
        self.weight.data.uniform_(-stdv,stdv)
        if self.bias_flag:
            nn.init.constant_(self.bias,0.0)

    def forward(self, input, order=None):
        return nn.functional.conv2d(input, self.quantize_weight(self.weight), self.bias, self.stride, self.padding, self.dilation, self.groups)

class QuantConv2d_WithBN(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=False,k_bit = 32,mode = 'linear'):
        super(QuantConv2d_WithBN,self).__init__()
        self.k_bits = k_bit
        self.momentum = 3e-4
        self.epsilon = 1e-3
        self.in_channels =  in_channels
        self.out_channels = out_channels
        self.kernel_size = _pair(kernel_size)
        self.quantize_weight = weight_quantize(self.k_bit,mode =mode)
        self.conv =nn.Conv2d(in_channels,out_channels,kernel_size=kernel_size,
                    stride = stride,bias = bias,padding=padding,dilation= dilation,
                    groups=groups)
        self.bn = nn.BatchNorm2d(out_channels,track_running_stats=True,momentum = self.momentum,eps=self.epsilon)
        self.convout = None
        self.bnout = None
        self.mean = None
        self.var = None
        self.beta = None
        self.gamma = None
        self.stand = None
        self.d = None
        self.folded_weight = None
        self.weight_q = None

        self.reset_parameters()
    
    def reset_parameters(self):
        n = self.in_channels
        for k in self.kernel_size:
            n *= k
        stdv = 1. / math.sqrt(n)
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def reset_parameter(self):
        nn.init.kaiming_normal_(self.conv.weight)
        nn.init.constant_(self.bn.weight,1.0)
        nn.init.constant_(self.bn.bias,0.0)
        nn.init.constant_(self.bn.running_mean,0.0)
        nn.init.constant_(self.bn.running_var,1.0)

    def forward(self,input,order = None):
        if self.training:
            self.convout = self.conv(input)
            self.bnout = self.bn(self.convout) 

            self.mean = self.bn.running_mean
            self.var = self.bn.running_var
        else:
            self.mean = self.bn.running_mean
            self.var = self.bn.running_var
        
        self.beta = self.bn.bias
        self.gamma = self.bn.weight
        self.stand = self.gamma / torch.sqrt(self.var + self.bn.eps)
        self.d = self.beta - self.mean * self.stand
        self.stand = self.stand[:,None,None,None]
        self.folded_weight = self.conv.weight * self.stand
        self.weight_q = self.quantize_weight(self.folded_weight)
        return nn.functional.conv2d(input, self.weight_q,self.d, self.conv.stride,
                        self.conv.padding, self.conv.dilation, self.conv.groups)

def conv3x3(in_channels, out_channels,kernel_size=3,stride=1,padding =1,bias= False):
    "3x3 convolution with padding"
    return nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=padding, bias=bias)

def quant_conv3x3(in_channels, out_channels,kernel_size=3,padding = 1,stride=1,mode='linear',k_bits=32,bias = False):
    Conv = QuantConv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride = stride,padding=padding ,mode=mode,k_bits=k_bits,bias = bias)
    return Conv

def quant_conv3x3_withbn(in_channels, out_channels, stride=1,mode='linear',k_bits=32):
    Conv= QuantConv2d_WithBN(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride = stride,padding=1 ,mode=mode,k_bit=k_bits)
    return Conv

def quant_linear(in_channels,out_channels,mode='linear',k_bits=32):
    Linear = QuantLinear(in_channels,out_channels,mode=mode,k_bits=k_bits)
    return Linear

def quant_linear_withbn(in_channels,out_channels,mode='linear',k_bits=32):
    Linear = QuantLinear_WithBN(in_channels,out_channels,bias=False,mode=mode,k_bits=k_bits)
    return Linear