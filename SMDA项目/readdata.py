# -*- coding: utf-8 -*-
"""
Created on Thu Mar 21 15:20:35 2019

@author: Liqi|leah0o
"""
import numpy as np 
import xlrd
import scipy.io as io
from sklearn.preprocessing import normalize as norm
from scipy.fftpack import fft
from os.path import splitext
 

def shuru(data):
    """
    shuru即输入
    读取原始数据
    """
    datatype=splitext(data)[1]  # 读取后缀名

    if datatype =='.xlsx':
        excel=xlrd.open_workbook(data)
        sheet=excel.sheets()[1]
        data=sheet.col_values(0)

    if datatype == '.mat':
        matdata=io.loadmat(data)
        fliter_=filter(lambda x: 'DE_time' in x, matdata.keys())  # 过滤排序，判断的DE_time是否在其中
        fliter_list = [item for item in fliter_]
        idx=fliter_list[0]  # 索引
        data=matdata[idx][:, 0]
    return data

def meanstd(data):    
    """
    to -1~1不用了，还是用sklearn的比较方便
    """
    for i in range(len(data)):
        datamean=np.mean(data[i])
        datastd=np.std(data[i])
        data[i]=(data[i]-datamean)/datastd
    
    return data

def sampling(data_this_doc,num_each,sample_lenth):
    """
    采样样本
    input:
        文件地址
        训练集的数量
        采样的长度
        故障的数量
    output:
        采样完的数据
    shuru->取长度->归一化
    ------
    note：采用的normalization 真这个话，除以l2范数
    """
    
    temp=shuru(data_this_doc)

    idx = np.random.randint(0, len(temp)-sample_lenth*2, num_each)
    temp_sample=[]
    for i in range(num_each):
        time=temp[idx[i]:idx[i]+sample_lenth*2]
        fre=abs(fft(time))[0:sample_lenth]#傅立叶变换
        temp_sample.append(fre) 
            

    temp_sample=np.array(temp_sample)
    temp_sample=norm(temp_sample)#正则化
    
    return temp_sample

class readdata():
    '''
    连接数据集、连接标签、输出
    '''
    def __init__(self,data_doc,num_each=400,ft=2,sample_lenth=1024):
        self.data_doc=data_doc
        ###特殊的再计算
        self.num_train=num_each*ft###
        
        self.ft=ft
        self.sample_lenth=sample_lenth
        self.row=num_each
    
    def concatx(self):
        """
        连接多个数据
        暂且需要有多少数据写多少数据
        """
        data=np.zeros((self.num_train,self.sample_lenth))
        for i,data_this_doc in enumerate(self.data_doc):
            data[0+i*self.row:(i+1)*self.row]=sampling(data_this_doc,self.row,self.sample_lenth)#?
        return data

    def labelling(self):   
        """
        根据样本数和故障类型生成样本标签
        one_hot
        """   
        label=np.zeros((self.num_train,self.ft))
        for i in range(self.ft):

            label[0+i*self.row:self.row+i*self.row,i]=1

        return label
       
    def output(self):
        '''
        输出数据集的数据和标签
        '''
        data=self.concatx()       
        label=self.labelling()
        size=int(float(self.sample_lenth)**0.5)
        data=data.astype('float32').reshape(self.num_train,1,size,size)
        label=label.astype('float32')
        return data,label
    

def dataset(train_data_name,data_name='sets',num_each=400,sample_lenth=1024,test_rate=0.5):
    '''
    根据特定的数据集构建，根据自己的需要修改路径
    '''
    test_data_name=train_data_name

    if test_rate==0:
        testingset=readdata(test_data_name,
                             ft=len(train_data_name),
                             num_each=num_each,
                             sample_lenth=sample_lenth)
        
        
        x_test,y_test=testingset.output()
        io.savemat(data_name,{'x_test': x_test,'y_test': y_test,})
        return x_test,y_test
    else:
        
        trainingset=readdata(train_data_name,
                             ft=len(train_data_name),
                             num_each=num_each,
                             sample_lenth=sample_lenth)
        
        testingset=readdata(test_data_name,
                             ft=len(train_data_name),
                             num_each=int(num_each*test_rate),
                             sample_lenth=sample_lenth)
        
        x_train,y_train=trainingset.output()
        x_test,y_test=testingset.output()
        io.savemat(data_name,{'x_train': x_train,'y_train': y_train,'x_test': x_test,'y_test': y_test,})
        return x_train,y_train,x_test,y_test
if __name__ == "__main__":
#%%
    '''
    SBDS 10 classification under various load
    '''
    SBDS_0K_10=['/home/c/liki/EGAN/自家试验台数据/002/4.xlsx',       
                      '/home/c/liki/EGAN/自家试验台数据/010/4.xlsx',
                      '/home/c/liki/EGAN/自家试验台数据/029/4.xlsx',
                      '/home/c/liki/EGAN/自家试验台数据/053/4.xlsx',
                      '/home/c/liki/EGAN/自家试验台数据/014/4.xlsx',
                      '/home/c/liki/EGAN/自家试验台数据/037/4.xlsx',
                      '/home/c/liki/EGAN/自家试验台数据/061/4.xlsx',
                      '/home/c/liki/EGAN/自家试验台数据/006/4.xlsx',
                      '/home/c/liki/EGAN/自家试验台数据/033/4.xlsx',
                      '/home/c/liki/EGAN/自家试验台数据/057/4.xlsx'
                      ]
    SBDS_1K_10=['/home/c/liki/EGAN/自家试验台数据/003/4.xlsx',       
                      '/home/c/liki/EGAN/自家试验台数据/011/4.xlsx',
                      '/home/c/liki/EGAN/自家试验台数据/030/4.xlsx',
                      '/home/c/liki/EGAN/自家试验台数据/054/4.xlsx',
                      '/home/c/liki/EGAN/自家试验台数据/015/4.xlsx',
                      '/home/c/liki/EGAN/自家试验台数据/038/4.xlsx',
                      '/home/c/liki/EGAN/自家试验台数据/062/4.xlsx',
                      '/home/c/liki/EGAN/自家试验台数据/007/4.xlsx',
                      '/home/c/liki/EGAN/自家试验台数据/034/4.xlsx',
                      '/home/c/liki/EGAN/自家试验台数据/058/4.xlsx'
                      ]
    SBDS_2K_10=['/home/c/liki/EGAN/自家试验台数据/004/4.xlsx',       
                      '/home/c/liki/EGAN/自家试验台数据/012/4.xlsx',
                      '/home/c/liki/EGAN/自家试验台数据/031/4.xlsx',
                      '/home/c/liki/EGAN/自家试验台数据/055/4.xlsx',
                      '/home/c/liki/EGAN/自家试验台数据/016/4.xlsx',
                      '/home/c/liki/EGAN/自家试验台数据/039/4.xlsx',
                      '/home/c/liki/EGAN/自家试验台数据/063/4.xlsx',
                      '/home/c/liki/EGAN/自家试验台数据/008/4.xlsx',
                      '/home/c/liki/EGAN/自家试验台数据/035/4.xlsx',
                      '/home/c/liki/EGAN/自家试验台数据/059/4.xlsx'
                      ]
    SBDS_3K_10=['/home/c/liki/EGAN/自家试验台数据/005/4.xlsx',       
                      '/home/c/liki/EGAN/自家试验台数据/013/4.xlsx',
                      '/home/c/liki/EGAN/自家试验台数据/032/4.xlsx',
                      '/home/c/liki/EGAN/自家试验台数据/056/4.xlsx',
                      '/home/c/liki/EGAN/自家试验台数据/017/4.xlsx',
                      '/home/c/liki/EGAN/自家试验台数据/040/4.xlsx',
                      '/home/c/liki/EGAN/自家试验台数据/064/4.xlsx',
                      '/home/c/liki/EGAN/自家试验台数据/009/4.xlsx',
                      '/home/c/liki/EGAN/自家试验台数据/036/4.xlsx',
                      '/home/c/liki/EGAN/自家试验台数据/060/4.xlsx'
                      ]
    '''
    CWRU 10 classification under various load
    '''
        
    CWRU_0hp_10=['/home/c/liki/基本框架/CWRUDS/97.mat', # 正常
            '/home/c/liki/基本框架/CWRUDS/105.mat',  # 0.17mm 内圈
            '/home/c/liki/基本框架/CWRUDS/118.mat',  # 0.17mm 滚动体
            '/home/c/liki/基本框架/CWRUDS/130.mat',  # 外圈
            '/home/c/liki/基本框架/CWRUDS/169.mat',  # 0.35mm
            '/home/c/liki/基本框架/CWRUDS/185.mat',
            '/home/c/liki/基本框架/CWRUDS/197.mat',
            '/home/c/liki/基本框架/CWRUDS/209.mat',  # 0.53mm
            '/home/c/liki/基本框架/CWRUDS/222.mat',
            '/home/c/liki/基本框架/CWRUDS/234.mat']

    CWRU_1hp_10=['/home/c/liki/基本框架/CWRUDS/98.mat',
            '/home/c/liki/基本框架/CWRUDS/106.mat',
            '/home/c/liki/基本框架/CWRUDS/119.mat',
            '/home/c/liki/基本框架/CWRUDS/131.mat',
            '/home/c/liki/基本框架/CWRUDS/170.mat',
            '/home/c/liki/基本框架/CWRUDS/186.mat',
            '/home/c/liki/基本框架/CWRUDS/198.mat',
            '/home/c/liki/基本框架/CWRUDS/210.mat',
            '/home/c/liki/基本框架/CWRUDS/223.mat',
            '/home/c/liki/基本框架/CWRUDS/235.mat']
    CWRU_2hp_10=['/home/c/liki/基本框架/CWRUDS/99.mat',
            '/home/c/liki/基本框架/CWRUDS/107.mat',
            '/home/c/liki/基本框架/CWRUDS/120.mat',
            '/home/c/liki/基本框架/CWRUDS/132.mat',
            '/home/c/liki/基本框架/CWRUDS/171.mat',
            '/home/c/liki/基本框架/CWRUDS/187.mat',
            '/home/c/liki/基本框架/CWRUDS/199.mat',
            '/home/c/liki/基本框架/CWRUDS/211.mat',
            '/home/c/liki/基本框架/CWRUDS/224.mat',
            '/home/c/liki/基本框架/CWRUDS/236.mat']
    CWRU_3hp_10=['/home/c/liki/基本框架/CWRUDS/100.mat',
            '/home/c/liki/基本框架/CWRUDS/108.mat',
            '/home/c/liki/基本框架/CWRUDS/121.mat',
            '/home/c/liki/基本框架/CWRUDS/133.mat',
            '/home/c/liki/基本框架/CWRUDS/172.mat',
            '/home/c/liki/基本框架/CWRUDS/188.mat',
            '/home/c/liki/基本框架/CWRUDS/200.mat',
            '/home/c/liki/基本框架/CWRUDS/212.mat',
            '/home/c/liki/基本框架/CWRUDS/225.mat',
            '/home/c/liki/基本框架/CWRUDS/237.mat']
    
    '''
    SBDS 4 classification under different load
    '''    
    SBDS_0K_4_06=['/home/c/liki/EGAN/自家试验台数据/002/4.xlsx',
                 '/home/c/liki/EGAN/自家试验台数据/053/4.xlsx',
                 '/home/c/liki/EGAN/自家试验台数据/061/4.xlsx',
                 '/home/c/liki/EGAN/自家试验台数据/057/4.xlsx'
            ]
    SBDS_1K_4_06=['/home/c/liki/EGAN/自家试验台数据/003/4.xlsx',
                 '/home/c/liki/EGAN/自家试验台数据/054/4.xlsx',
                 '/home/c/liki/EGAN/自家试验台数据/062/4.xlsx',
                 '/home/c/liki/EGAN/自家试验台数据/058/4.xlsx'
            ]
    SBDS_2K_4_06=['/home/c/liki/EGAN/自家试验台数据/004/4.xlsx',
                 '/home/c/liki/EGAN/自家试验台数据/055/4.xlsx',
                 '/home/c/liki/EGAN/自家试验台数据/063/4.xlsx',
                 '/home/c/liki/EGAN/自家试验台数据/059/4.xlsx'
            ]
    SBDS_3K_4_06=['/home/c/liki/EGAN/自家试验台数据/005/4.xlsx',
                 '/home/c/liki/EGAN/自家试验台数据/056/4.xlsx',
                 '/home/c/liki/EGAN/自家试验台数据/064/4.xlsx',
                 '/home/c/liki/EGAN/自家试验台数据/060/4.xlsx'
            ]
    '''
    CWRU 4 classification under different load
    '''
    CWRU_0hp_4=['/home/c/liki/基本框架/CWRUDS/97.mat',
                '/home/c/liki/基本框架/CWRUDS/209.mat',
                '/home/c/liki/基本框架/CWRUDS/222.mat',
                '/home/c/liki/基本框架/CWRUDS/234.mat']
    
    CWRU_1hp_4=['/home/c/liki/基本框架/CWRUDS/98.mat',
                '/home/c/liki/基本框架/CWRUDS/210.mat',
                '/home/c/liki/基本框架/CWRUDS/223.mat',
                '/home/c/liki/基本框架/CWRUDS/235.mat']
    
    CWRU_2hp_4=['/home/c/liki/基本框架/CWRUDS/99.mat',
                '/home/c/liki/基本框架/CWRUDS/211.mat',
                '/home/c/liki/基本框架/CWRUDS/224.mat',
                '/home/c/liki/基本框架/CWRUDS/236.mat']
    
    CWRU_3hp_4=['/home/c/liki/基本框架/CWRUDS/100.mat',
                '/home/c/liki/基本框架/CWRUDS/212.mat',
                '/home/c/liki/基本框架/CWRUDS/225.mat',
                '/home/c/liki/基本框架/CWRUDS/237.mat']
    
#%%
    dataset(CWRU_3hp_10,data_name='datasets/CWRU_3hp_10')