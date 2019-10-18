from __future__ import absolute_import
from __future__ import unicode_literals
from __future__ import print_function
from __future__ import division

import torch
import torch.nn as nn
from torch.autograd import Variable
from functools import reduce
import operator
import pdb

count_ops = []
count_params = []
# oritin_flops, origin_params = 125485706.0, 848954.0
origin_flops = {'resnet_18': 555422730, 'resnet_50': 1297829898.00}
origin_params = {'resnet_18': 11164362, 'resnet_50': 23467722.00}

def get_ratio(origin, pruned):
    ratio = pruned / origin
    return ratio * 100.

def get_num_gen(gen):
    return sum(1 for x in gen)

def is_pruned(layer):
    try:
        layer.mask
        return True
    except AttributeError:
        return False

def is_leaf(model):
    
    return get_num_gen(model.children()) == 0 and 'ReLU' not in str(model) and 'Shorcut' not in str(model)

def get_layer_info(layer):
    layer_str = str(layer)
    type_name = layer_str[:layer_str.find('(')].strip()
    return type_name

def get_layer_param(model):
    return sum([reduce(operator.mul, i.size(), 1) for i in model.parameters()])

def measure_layer(layer, x):

    global count_ops, count_params
    delta_ops = 0
    delta_params = 0
    multi_add = 1
    type_name = get_layer_info(layer)
    # print(type_name)

    ### ops_conv
    if type_name in ['Conv2d','AULMConv2d']:
        out_h = int((x.size()[2] + 2 * layer.padding[0] - layer.kernel_size[0]) /
                    layer.stride[0] + 1)
        out_w = int((x.size()[3] + 2 * layer.padding[1] - layer.kernel_size[1]) /
                    layer.stride[1] + 1)
        delta_ops = layer.in_channels * layer.out_channels * layer.kernel_size[0] *  \
                layer.kernel_size[1] * out_h * out_w / layer.groups * multi_add
        delta_params = get_layer_param(layer)
        # print(layer,  delta_params)

    ### ops_linear
    elif type_name in ['Linear']:
        weight_ops = layer.weight.numel() * multi_add
        bias_ops = layer.bias.numel()
        delta_ops = x.size()[0] * (weight_ops + bias_ops)
        delta_params = get_layer_param(layer)

    elif type_name in ['MeanShift', 'ReLU','ShortCut','BatchNorm2d', 'pact_activation_quantize', 'weight_quantize', 'Dropout', 'AdaptiveAvgPool2d', 'AvgPool2d', 'MaxPool2d', 'Mask', 'ChannelSelectLayer', 'LambdaLayer', 'Sequential', 'ReLU6', 'PixelShuffle']:
        
        return 
    ### unknown layer type
    else:
        # print(layer)
        raise TypeError('unknown layer type: %s' % type_name)

    count_ops.append(delta_ops)
    count_params.append(delta_params)
    return


def measure_model(model, args, device, C, H, W, scale=4):
    global count_ops, count_params
    count_ops = []
    count_params = []
    data = Variable(torch.zeros(1, C, H, W)).to(device)
    model = model.to(device)
    model.eval()

    def should_measure(x):
        # print('in should_measure',x)
        return is_leaf(x)

    def modify_forward(model):
        for child in model.children():
            if should_measure(child):
                # print('in modify_forward', child)
                def new_forward(m):
                    def lambda_forward(x):
                        # print(m)
                        measure_layer(m, x)
                        return m.old_forward(x)
                    return lambda_forward
                child.old_forward = child.forward
                child.forward = new_forward(child)
            else:
                modify_forward(child)

    def restore_forward(model):
        for child in model.children():
            # leaf node
            if is_leaf(child) and hasattr(child, 'old_forward'):
                child.forward = child.old_forward
                child.old_forward = None
            else:
                restore_forward(child)

    with torch.no_grad():
        modify_forward(model)
        model.forward(data, scale)
        restore_forward(model)

    # ratio_flops = get_ratio(origin_flops[f'{args.arch}_{args.num_layers}'], count_ops)
    # ratio_params = get_ratio(origin_params[f'{args.arch}_{args.num_layers}'], count_params)
    return count_ops, count_params
