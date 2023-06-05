# -*- coding: utf-8 -*-
"""
Created on Tue Oct  6 00:00:52 2020

@author: 98133
"""

import numpy as np
import torch


def dim_permute(h):
    if len(h.size())>2:
        h=h.permute(1,0,2,3).contiguous()
        h = h.view(h.size(0), -1)
    else:
        h=h.permute(1,0).contiguous()
        h = h.view(h.size(0),-1)
    return h


def compute_l2_norm(h, subtract_mean=False):
    h = dim_permute(h)
    N = (h.size(1))
    if subtract_mean:
        mn = (h).mean(dim=1, keepdim=True)
        h = h-mn

    l2_norm = (h**2).sum()
    return torch.sqrt(l2_norm)


def correlation_reg(hid, targets, within_class=True, subtract_mean=True):
    """
    hid：源域的输入特征
    targets：目标域的输出特征
    """
    norm_fn = compute_l2_norm
    if within_class:
        targets = targets.cpu().detach().numpy()
        uniq = np.unique(targets)
        reg_=0
        for u in uniq:
            idx = np.where(targets==u)[0]
        
            norm = norm_fn(hid[idx], subtract_mean=subtract_mean)
            reg_ += (norm)**2
    else:
        norm = norm_fn(hid, subtract_mean=subtract_mean)
        reg_ = (norm)**2
    return reg_