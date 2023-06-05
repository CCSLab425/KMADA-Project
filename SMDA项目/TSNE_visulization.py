# -*- coding: utf-8 -*-
"""
Created on Sun Dec 16 21:22:51 2018

@author: 李奇
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from os.path import splitext
from scipy.io import loadmat


def data_reader(datadir,gpu=True):
    '''
    read data from mat or other file after readdata.py   
    '''
    datatype=splitext(datadir)[1]
    if datatype=='.mat':

        print('datatype is mat')

        data=loadmat(datadir)
        
        x_train=data['x_train']
        x_test=data['x_test']     
        y_train=data['y_train']     
        y_test=data['y_test']
    if datatype=='':
        pass

    return x_train,y_train,x_test,y_test


def pca_(matname):
    def to_1dim(array):
        return np.mean(array,axis=1)

    sfis=loadmat(matname)['sfis']
    sfit=loadmat(matname)['sfit']
    tfis=loadmat(matname)['tfis']
    tfit=loadmat(matname)['tfit']

    sfis=to_1dim(sfis)
    sfit=to_1dim(sfit)
    tfis=to_1dim(tfis)
    tfit=to_1dim(tfit)

    D2shapes = sfis.shape[1]*sfis.shape[2]
    pca = PCA(n_components=2)
    pca_resultss = pca.fit_transform(sfis.reshape([-1, D2shapes]))
    pca_resulttt = pca.fit_transform(tfit.reshape([-1, D2shapes]))
    
    pca_dic = {'s_s':pca_resultss,
               's_t':pca_resultst,
               't_s':pca_resultts,
               't_t':pca_resulttt}

    return pca_dic

        
def tSNE_(matname):

    sfis=loadmat(matname)['Fs'].reshape(2000,-1)
    tfit=loadmat(matname)['Ft'].reshape(2000,-1)
    sfis_plus_tfit = np.append(sfis,tfit,axis=0)
    
    tSNE = TSNE(n_components=2,n_iter=400)
    tSNE_results = tSNE.fit_transform(sfis_plus_tfit)

    return tSNE_results


def plot2D(dic,setname='SBDS',save_dir='plot_dir/',TLname='default'):
        
    tSNEx=dic[:,0]
    tSNEy=dic[:,1]

    clist = ['r','lightsalmon','gold','yellow','palegreen',
             'lightseagreen','lightblue','slateblue','violet','purple']
    labelCWRU = ['NOR','I07','I14','I21',
             'B07','B14','B21',
             'O07','O14','O21'
             ]
    labelSBDB = ['Nor','I02','I04','I06',
     'B02','B04','B06',
     'O02','O04','O06'
     ]
    if setname=='CWRU':
        label=labelCWRU
    elif setname=='SBDB':
        label=labelSBDB
    
    markerlist=['.',',','o','v','^','<','>','D','*','h','x','+']
    
#    num = 202
    num = 602
    
    sample = 40
    target_idx = 2000
    target_idx2 = 4000

    target_idx3 = 6000

# for 4
    for i in range(4):
        
        
        plt.scatter(tSNEx[0+i*num:sample+i*num],tSNEy[0+i*num:sample+i*num],
                    s=100,marker=markerlist[i],c='none',edgecolor=clist[-3],label='S0'+label[i])

        plt.scatter(tSNEx[target_idx+i*num:target_idx+sample+i*num],tSNEy[target_idx+i*num:target_idx+sample+i*num],
                    s=100,marker=markerlist[i],c='none',edgecolor=clist[1],label='S1'+label[i])

        plt.scatter(tSNEx[target_idx2+i*num:target_idx2+sample+i*num],tSNEy[target_idx2+i*num:target_idx2+sample+i*num],
                    s=100,marker=markerlist[i],c='none',edgecolor=clist[3],label='S2'+label[i])

        plt.scatter(tSNEx[target_idx3+i*num:target_idx3+sample+i*num],tSNEy[target_idx3+i*num:target_idx3+sample+i*num],
                    s=100,marker=markerlist[i],c='none',edgecolor=clist[5],label='T'+label[i])

    plt.xticks([])
    plt.yticks([])
    plt.savefig(save_dir+TLname+'.svg',format='svg')
    plt.show()

    
if __name__ == '__main__': 

    DIR = 'SBDSsave_dir/feature2/'
    import os
    filelist = os.listdir(DIR)
    for file in filelist:
        
        tSNEresult = tSNE_(DIR+file)
        plot2D(tSNEresult,setname='SBDB',save_dir=DIR,TLname=file)
        

        
                

