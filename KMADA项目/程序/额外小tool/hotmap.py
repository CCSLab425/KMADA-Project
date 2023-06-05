
from __future__ import division

import  numpy as np

import matplotlib.pyplot as plt

from matplotlib.ticker import MultipleLocator

 

def plotCM(classes, matrix, savname):

    """classes: a list of class names"""

    # Normalize by row
    matrix = np.array(matrix ).T
    matrix = matrix.astype(np.float)

#    linesum = matrix.sum(1)
#
#    linesum = np.dot(linesum.reshape(-1, 1), np.ones((1, matrix.shape[1])))
#
#    matrix /= linesum

    # plot

    plt.switch_backend('agg')

    fig = plt.figure()

    ax = fig.add_subplot(111)
    
    cm = plt.cm.get_cmap('RdYlBu')
    
    cax = ax.matshow(matrix,cmap=cm)

    fig.colorbar(cax)

    ax.xaxis.set_major_locator(MultipleLocator(1))

    ax.yaxis.set_major_locator(MultipleLocator(1))

    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):

            ax.text(i, j, str('%.4f' % (matrix[j, i])), va='center', ha='center')

    ax.set_xticklabels([''] + classes)#, rotation=90

    ax.set_yticklabels([''] + classes)

    #save

    plt.savefig(savname+'.svg',format='svg')
    plt.savefig(savname+'.png',format='png')
    plt.plot()
if __name__=='__main__':
	classes = ['0.0001','0.0005','0.001','0.005','0.01']
	matrix =  [[0.7995	,1.000,0.99975,0.1075,0.1],
				[0.69975,	0.8	,0.7,0.1,0.1],
				[0.799,0.89375,0.9,0.1	,0.1],
				[0.69875,	0.89975,0.7895,0.094	,0.1],
				[0.8995,1.000,0.6995,0.1,0.1]]
				
	savname = 'lrCW'
	plotCM(classes, matrix, savname)