import os
import torch.nn as nn
import torch
import numpy as np
import scipy.misc

import matplotlib.pyplot as plt

#调用cuda，未使用
def make_cuda(tensor):
    """Use CUDA if it's available."""
    if torch.cuda.is_available():
        tensor = tensor.cuda()
    return tensor



# 返回迭代对象中的每个元素，未使用
def loop_iterable(iterable):
    while True:
        yield from iterable



#打印网络结构
def print_network(net):
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    print(net)
    print('Total number of parameters: %d' % num_params)


# 随机初始化
def initialize_weights(net):
    for m in net.modules():
        if isinstance(m, nn.Conv2d):
            
            nn.init.xavier_normal_(m.weight, gain=1)
            m.bias.data.zero_()
        elif isinstance(m, nn.ConvTranspose2d):
            nn.init.xavier_normal_(m.weight, gain=1)
            m.bias.data.zero_()
        elif isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight, gain=1)
            m.bias.data.zero_()