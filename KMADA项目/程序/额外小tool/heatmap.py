# -*- coding: utf-8 -*-
"""
Created on Wed Dec 11 20:24:31 2019

@author: 李奇
"""

import scipy.io as spio
import numpy as np
import pandas as pd
from pandas import DataFrame,Series
import seaborn as sns
import os
import matplotlib.pyplot as plt
import time
import random
import h5py
#df = pd.DataFrame(data,index = range(data.shape[0]),columns=[
#'data1','data2','label'])

corrmat= [[1,0,0,0,0,0,0,0,0,0],
[0,1,0,0,0,0,0,0,0,0],
[0,0,1,0,0,0,0,0,0,0],
[0,0,0,1,0,0,0,0,0,0],
[0,0,0,0,0.975,0.005,0,0,0,0],
[0,0,0,0,0.025,0.995,0,0,0,0],
[0,0,0,0,0,0,1,0,0,0],
[0,0,0,0,0,0,0,1,0,0],
[0,0,0,0,0,0,0,0,1,0],
[0,0,0,0,0,0,0,0,0,1]]
print(corrmat)
font1 = {'family': 'Times New Roman',
         'weight': 'black',
         'size': 20,
         }


cmap = sns.cubehelix_palette(start = 1.5, rot = 3, gamma=0.8, as_cmap = True)

xticks=['Nor','O02','I02','B02','O04','I04','B04','O06','I06','B06']
yticks=['Nor','O02','I02','B02','O04','I04','B04','O06','I06','B06']

sns.set_context({"figure.figsize": (12, 12)})
#plt.xlabel(font1)
#plt.ylabel(font1)
ax = sns.heatmap(corrmat,annot=True,annot_kws=font1, fmt='.3g',vmax=1, square=True,cmap='Blues')

ax.set_xlabel('Predicted labels',font1)
ax.set_ylabel('Actual labels',font1)
ax.set_xticklabels(xticks,font1)
ax.set_yticklabels(yticks,font1)
bottom, top = ax.get_ylim()
ax.set_ylim(bottom + 0.5, top - 0.5)

plt.savefig('hetmap_source2'+'.svg')
plt.show()