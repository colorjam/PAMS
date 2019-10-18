
import torch
import torch.nn as nn
import torch.nn.parallel as P
from model.quantize_ops import quant_conv3x3, activation_quantize, quant_conv3x3_withbn,pact_activation_quantize,QuantConv2d
import pdb
__all__ = ['rdn']

def make_model(args, parent=False):
    return RDN(args)

class ShortCut(nn.Module):
    def __init__(self):
        super(ShortCut, self).__init__()

    def forward(self, input):
        return input

class Pact_RDB_Conv(nn.Module):
    def __init__(self, inChannels, growRate, kSize=3,k_bits=32,mode = 'max',version = 3,name = None):
        super(Pact_RDB_Conv, self).__init__()
        Cin = inChannels
        G  = growRate

        self.k_bits = k_bits
        self.mode = mode
        self.version = version

        self.conv = nn.Sequential(*[
            quant_conv3x3(Cin, G, kSize, padding=(kSize-1)//2, stride =1, mode = self.mode, k_bits= self.k_bits, bias = True),
            nn.ReLU()
        ])
        
        self.act1 = pact_activation_quantize(self.k_bits, self.mode, self.version, name)
        self.act2 = pact_activation_quantize(self.k_bits, self.mode, self.version, name)
        
    def forward(self, x):
        x1 = self.act1(x)
        out = self.act2(self.conv(x))

        return torch.cat((x1, out), 1)

class Pact_RDB_Conv_in(nn.Module):
    def __init__(self, inChannels, growRate, kSize=3, k_bits=32, mode = 'max', version = 3,name = None):
        super(Pact_RDB_Conv_in, self).__init__()
        Cin = inChannels
        G  = growRate

        self.k_bits = k_bits
        self.mode = mode
        self.version = version

        self.conv = nn.Sequential(*[
            quant_conv3x3(Cin, G, kSize, padding=(kSize-1)//2, stride = 1, mode = self.mode, k_bits= self.k_bits, bias = True),
            nn.ReLU()
        ])
        
        self.act = pact_activation_quantize(self.k_bits, self.mode, self.version, name)
        
    def forward(self, x, i):
        if i > 0:
            x = self.act(x)
        out = self.conv(x)

        return torch.cat((x, out), 1)

class Pact_RDB(nn.Module):
    def __init__(self, growRate0, growRate, nConvLayers, kSize=3, k_bits=32, mode = 'max', version = 3,name= None):
        super(Pact_RDB, self).__init__()
        G0 = growRate0
        G  = growRate
        C  = nConvLayers

        self.k_bits = k_bits
        self.mode = mode
        self.version = version

        convs = []
        for c in range(C):
            convs.append(Pact_RDB_Conv_in(G0 + c*G, G,kSize ,k_bits = self.k_bits,mode= self.mode,version = self.version,name = name))
        self.convs = nn.Sequential(*convs)
        
        self.act1 = pact_activation_quantize(self.k_bits, self.mode, self.version)
        self.act2 = pact_activation_quantize(self.k_bits, self.mode, self.version)

        # Local Feature Fusion
        self.LFF = QuantConv2d(in_channels=G0 + C*G, out_channels=G0, kernel_size=1, padding=0, mode=self.mode,k_bits=self.k_bits, stride=1, bias=True)

    def forward(self, x):
        x = self.act1(x)
        out = x
        for i, c in enumerate(self.convs):
            out = c(out, i)
        return self.LFF(self.act2(out)) + x

    @property
    def name(self):
        return 'rdb'

class RDN_PAMS(nn.Module):
    def __init__(self,args):
        super(RDN_PAMS, self).__init__()
        r = args.scale[0]
        G0 = args.G0
        kSize = args.RDNkSize
        self.mode = args.mode
        self.version = args.version
        # print(self.version)

        # number of RDB blocks, conv layers, out channels
        self.D, C, G = {
            'A': (20, 6, 32),
            'B': (16, 8, 64),
        }[args.RDNconfig]

        if not type([]) == type(args.k_bits):
            self.k_bits = [args.k_bits for _ in range(self.D)]
        else:
            self.k_bits = args.k_bits

        # Shallow feature extraction net
        self.SFENet1 = nn.Conv2d(args.n_colors, G0, kSize, padding=(kSize-1)//2, stride=1)
        self.SFENet2 = nn.Conv2d(G0, G0, kSize, padding=(kSize-1)//2, stride=1)
        # Redidual dense blocks and dense feature fusion
        self.RDBs = nn.ModuleList()
        for i in range(self.D):
            self.RDBs.append(
                Pact_RDB(growRate0 = G0, growRate = G, nConvLayers = C, mode = self.mode, k_bits=args.k_bits,version=self.version, name = self.name)
            )

        self.act = pact_activation_quantize(args.k_bits, self.mode, self.version)

        # Global Feature Fusion
        self.GFF = nn.Sequential(*[
            QuantConv2d(self.D * G0, G0, 1, padding=0, stride=1, mode=self.mode, k_bits=args.k_bits,bias=True),
            QuantConv2d(G0, G0, kSize, padding=(kSize-1)//2, stride=1,mode=self.mode, k_bits=args.k_bits,bias=True)
        ])

        # Up-sampling net
        if r == 2 or r == 3:
            self.UPNet = nn.Sequential(*[
                nn.Conv2d(G0, G * r * r, kSize, padding=(kSize-1)//2, stride=1),
                nn.PixelShuffle(r),
                nn.Conv2d(G, args.n_colors, kSize, padding=(kSize-1)//2, stride=1)
            ])
        elif r == 4:
            self.UPNet = nn.Sequential(*[
                nn.Conv2d(G0, G * 4, kSize, padding=(kSize-1)//2, stride=1),
                nn.PixelShuffle(2),
                nn.Conv2d(G, G * 4, kSize, padding=(kSize-1)//2, stride=1),
                nn.PixelShuffle(2),
                nn.Conv2d(G, args.n_colors, kSize, padding=(kSize-1)//2, stride=1)
            ])
        else:
            raise ValueError("scale must be 2 or 3 or 4.")

    def forward(self, x):
        f__1 = self.SFENet1(x)
        x  = self.SFENet2(f__1)
        RDBs_out = []
        for i in range(self.D):
            x = self.RDBs[i](x)
            RDBs_out.append(x)

        # pdb.set_trace()
        x = self.GFF(self.act(torch.cat(RDBs_out, 1)))
        x += f__1

        out = x 
        return self.UPNet(x), out

    @property
    def name(self):
        return 'rdn'
