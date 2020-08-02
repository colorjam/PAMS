
import math
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from model import common
from model.quant_ops import conv3x3
from model.quant_ops import pams_quant_act
from model.quant_ops import quant_conv3x3

class PAMS_ResBlock(nn.Module):
    def __init__(self, conv, n_feats, kernel_size, bias=False, 
                bn=False, act=nn.ReLU(False), res_scale=1, k_bits = 32, ema_epoch=1, name=None):

        super(PAMS_ResBlock, self).__init__()
        self.k_bits = k_bits

        self.quant_act1 = pams_quant_act(self.k_bits,ema_epoch=ema_epoch)
        self.quant_act2 = pams_quant_act(self.k_bits, ema_epoch=ema_epoch)
        self.quant_act3 = pams_quant_act(self.k_bits, ema_epoch=ema_epoch)

        self.shortcut = common.ShortCut()

        m = []
        for i in range(2):
            m.append(conv(n_feats, n_feats, kernel_size, k_bits=self.k_bits, bias=bias))
            if i == 0: 
                m.append(act) 
                m.append(self.quant_act2) 
            
        self.body = nn.Sequential(*m)
        self.res_scale = res_scale

    def forward(self, x):
        residual = self.quant_act1(self.shortcut(x))
        body = self.body(x).mul(self.res_scale)
        res = self.quant_act3(body)
        res += residual

        return res

class PMAS_EDSR(nn.Module):
    def __init__(self, args, conv=quant_conv3x3, bias = False, k_bits = 32):
        super(PMAS_EDSR, self).__init__()

        n_resblock = args.n_resblocks
        n_feats = args.n_feats
        kernel_size = 3 
        scale = args.scale[0]
        act = nn.ReLU(True)
        self.k_bits = args.k_bits
        
        self.sub_mean = common.MeanShift(args.rgb_range)
        self.add_mean = common.MeanShift(args.rgb_range, sign=1)

        # define head module
        m_head = [conv3x3(args.n_colors, n_feats, kernel_size,  bias=bias)]

        # define body module
        m_body = [
            PAMS_ResBlock(
                quant_conv3x3, n_feats, kernel_size, act=act, res_scale=args.res_scale, k_bits=self.k_bits, bias = bias, ema_epoch=args.ema_epoch
            ) for i in range(n_resblock)
        ]

        m_body.append(conv3x3(n_feats, n_feats, kernel_size, bias= bias))

        # define tail module
        m_tail = [
            common.Upsampler(conv3x3, scale, n_feats, act=False),
            nn.Conv2d(
                n_feats, args.n_colors, kernel_size,
                padding=(kernel_size//2)
            )
        ]
        
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
