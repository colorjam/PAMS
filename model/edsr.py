

import torch
import torch.nn as nn
import math
import pdb
import math
import torch.nn.functional as F
from torch.autograd import Variable
import time

from model.quantize_ops import activation_quantize,quant_conv3x3,conv3x3,pact_activation_quantize

__all__ =['edsr']

def default_conv(in_channels, out_channels, kernel_size,
                mode= 'max', k_bits = 32,version = 1,name='res',bias = False):
    return quant_conv3x3(in_channels, out_channels, kernel_size,stride=1,mode=mode,k_bits=k_bits,bias = bias)

class MeanShift(nn.Conv2d):
    def __init__(self, rgb_range, rgb_mean, rgb_std, sign=-1):
        super(MeanShift, self).__init__(3, 3, kernel_size=1)
        std = torch.Tensor(rgb_std)
        self.weight.data = torch.eye(3).view(3, 3, 1, 1)
        self.weight.data.div_(std.view(3, 1, 1, 1))
        self.bias.data = sign * rgb_range * torch.Tensor(rgb_mean)
        self.bias.data.div_(std)
        self.requires_grad = False

class ShortCut(nn.Module):
    def __init__(self):
        super(ShortCut, self).__init__()

    def forward(self, input):
        return input

class BasicResBlock(nn.Module):
    def __init__(self, conv, n_feats, kernel_size,bias=False, 
                bn=False, act=nn.ReLU(False), res_scale=1,mode= 'max',k_bits = 32,version =1,name=None,num = None):
        super(BasicResBlock, self).__init__()
        self.mode = mode
        self.k_bits = k_bits
        self.version = version 
        self.num = num

        self.shortcut = ShortCut()
        
        m = []
        for i in range(2):
            m.append(conv3x3(n_feats,n_feats,kernel_size,1,padding=kernel_size//2,bias=bias))
            if i == 0: 
                m.append(act)
            
        self.body = nn.Sequential(*m)
        self.res_scale = res_scale

    def forward(self, x):
        residual = self.shortcut(x)
        res = self.body(x).mul(self.res_scale)
        res += residual
        return res


class ResBlock(nn.Module):
    def __init__(self, conv, n_feats, kernel_size,bias=False, 
                bn=False, act=nn.ReLU(False), res_scale=1, mode= 'max',k_bits=32,version=1,name=None,num = None):
        super(ResBlock, self).__init__()
        self.mode = mode
        self.k_bits = k_bits
        self.version = version 
        self.num = num

        self.activation1 = activation_quantize(self.k_bits,mode=mode,version=version,name=name)
        self.activation2 = activation_quantize(self.k_bits,mode=mode,version=version,name=name)
        self.activation3 = activation_quantize(self.k_bits,mode=mode,version=version,name=name)

        self.shortcut = ShortCut()
        m = []
        for i in range(2):
            m.append(conv(n_feats,n_feats,kernel_size,mode=mode,k_bits=self.k_bits,bias = bias))
            if i == 0: 
                m.append(act)
                m.append(self.activation2)
            
        self.body = nn.Sequential(*m)
        self.res_scale = res_scale

    def forward(self, x):
        self.activation1.k_bits = self.k_bits
        self.activation3.k_bits = self.k_bits
        for i in [0,2,3]:
            self.body[i].k_bits = self.k_bits
        residual = self.activation1(self.shortcut(x))
        res = self.activation3(self.body(x).mul(self.res_scale))
        res += residual
        return res


class PAMS_ResBlock(nn.Module):
    def __init__(self, conv, n_feats, kernel_size,bias=False, 
                bn=False, act=nn.ReLU(False), res_scale=1,mode= 'max',k_bits = 32,version =1,name=None,num=None,ema_epoch=1):

        super(PAMS_ResBlock, self).__init__()
        self.mode = mode
        self.k_bits = k_bits
        self.version = version 

        self.activation1 = pact_activation_quantize(self.k_bits,mode=mode,version=version, name=name, ema_epoch=ema_epoch)
        self.activation2 = pact_activation_quantize(self.k_bits,mode=mode,version=version, name=name,ema_epoch=ema_epoch)
        self.activation3 = pact_activation_quantize(self.k_bits,mode=mode,version=version, name=name,ema_epoch=ema_epoch)

        self.shortcut = ShortCut()

        m = []

        for i in range(2):
            if bn :
                m.append(quant_conv3x3_withbn(n_feats,n_feats,stride=1,mode=mode,k_bits=self.k_bits))
            else:
                m.append(conv(n_feats,n_feats,kernel_size,mode=mode,k_bits=self.k_bits,bias = bias)) # 0 / 3
            if i == 0: 
                m.append(act) # 1
                m.append(self.activation2) # 2
            
        self.body = nn.Sequential(*m)
        self.res_scale = res_scale

        self.activation1.k_bits = self.k_bits
        self.activation3.k_bits = self.k_bits

        for i in [0, 2, 3]:
            self.body[i].k_bits = self.k_bits

    def forward(self, x):
        residual = self.activation1(self.shortcut(x))
        body = self.body(x).mul(self.res_scale)
        res = self.activation3(body)
        res += residual

        return res

class Upsampler(nn.Sequential):
    def __init__(self, conv, scale, n_feats, bn=False, act=False, bias=True):

        m = []
        if (scale & (scale - 1)) == 0:    # Is scale = 2^n?
            for _ in range(int(math.log(scale, 2))):
                m.append(conv(n_feats, 4 * n_feats, 3, 1,padding =1, bias=bias))
                m.append(nn.PixelShuffle(2))
                if bn: m.append(nn.BatchNorm2d(n_feats))
                if act == 'relu':
                    m.append(nn.ReLU(True))
                elif act == 'prelu':
                    m.append(nn.PReLU(n_feats))

        elif scale == 3:
            m.append(conv(n_feats, 9 * n_feats, 3, bias=bias))
            m.append(nn.PixelShuffle(3))
            if bn: m.append(nn.BatchNorm2d(n_feats))

            if act == 'relu':
                m.append(nn.ReLU(True))
            elif act == 'prelu':
                m.append(nn.PReLU(n_feats))
        else:
            raise NotImplementedError

        super(Upsampler, self).__init__(*m)

class EDSR_PAMS(nn.Module):
    def __init__(self, args, conv=default_conv, bias = False, k_bits=None):
        super(EDSR_PAMS, self).__init__()
        n_resblock = args.n_resblocks
        n_feats = args.n_feats
        kernel_size = 3 
        scale = args.scale[0]
        act = nn.ReLU(True)
        self.mode = args.mode
        self.version = args.version
        self.k_bits = args.k_bits
        
        rgb_mean = (0.4488, 0.4371, 0.4040)
        rgb_std = (1.0, 1.0, 1.0)

        self.sub_mean = MeanShift(args.rgb_range, rgb_mean, rgb_std)
        
        # define head module
        m_head = [conv3x3(args.n_colors, n_feats, kernel_size,1, bias=bias)]
        # define body module
        m_body = [
            PAMS_ResBlock(
                conv, n_feats, kernel_size, act=act, res_scale=args.res_scale,
                mode=self.mode, k_bits=self.k_bits, version=self.version, name=self.name,num=i,bias = bias,ema_epoch=args.ema_epoch
            ) for i in range(n_resblock)
        ]
        m_body.append(conv3x3(n_feats, n_feats, kernel_size,bias = bias))

        # define tail module
        m_tail = [
            Upsampler(conv3x3, scale, n_feats, act=False),
            nn.Conv2d(
                n_feats, args.n_colors, kernel_size,
                padding=(kernel_size//2)
            )
        ]
        self.add_mean = MeanShift(args.rgb_range, rgb_mean, rgb_std, 1)

        self.head = nn.Sequential(*m_head)
        self.body = nn.Sequential(*m_body)
        self.tail = nn.Sequential(*m_tail)

    def forward(self, x):
        x = self.sub_mean(x)
        x = self.head(x)
        end = time.time()
        res = self.body(x)
        res += x
        out = res
        x = self.tail(res)
        x = self.add_mean(x)

        return x, out   
    
    @property
    def name(self):
        return 'edsr'