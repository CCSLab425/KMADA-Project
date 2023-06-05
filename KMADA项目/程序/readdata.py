# -*- coding: utf-8 -*-
"""
Created on Thu Mar 21 15:20:35 2019

@author: Liqi
"""
import numpy as np 
import xlrd
import scipy.io as io
from sklearn.preprocessing import normalize as norm
from scipy.fftpack import fft
from os.path import splitext
import pandas as pd
#from STFT import stft 


def shuru(data):
    """
    load data from xlsx
    """
    datatype=splitext(data)[1]

    if datatype =='.xlsx':
        excel=xlrd.open_workbook(data)
        sheet=excel.sheets()[1]
        data=sheet.col_values(0)
        
    if datatype =='.csv':
        data = pd.read_csv(data,engine='python')

        data = np.array(data)
        data = data.flatten()
        #data = data.reshape(1,-1)
        
    if datatype == '.mat':
        matdata=io.loadmat(data)
        fliter_=filter(lambda x: 'DE_time' in x, matdata.keys())
        fliter_list = [item for item in fliter_]
        idx=fliter_list[0]
        data=matdata[idx][:, 0]
        
    if datatype == '.txt':
        data=np.loadtxt(data)[:,1]
    
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

def sampling(data_this_doc,num_each,sample_lenth,spiltflag = 'train',spilt_ratio=0.7):
    """
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

   
    if spiltflag=='train':
        idx = np.random.randint(0, len(temp)*spilt_ratio-sample_lenth*2, num_each)
    if spiltflag=='test':
        idx = np.random.randint(len(temp)*spilt_ratio, len(temp)-sample_lenth*2, num_each)
        
    temp_sample=[]
    for i in range(num_each):
        time=temp[idx[i]:idx[i]+sample_lenth*2]
        fre=abs(fft(time))[0:sample_lenth]
        temp_sample.append(fre) 
            

    temp_sample=np.array(temp_sample)
    print(temp_sample.size)
    temp_sample=norm(temp_sample)
    
    return temp_sample

class readdata():
    '''
    连接数据集、连接标签、输出
    '''
    def __init__(self,data_doc,num_each=400,ft=4,sample_lenth=1024,spiltflag='train'):
        self.data_doc=data_doc
        ###特殊的再计算
        self.num_train=num_each*ft###
        
        self.ft=ft
        self.sample_lenth=sample_lenth
        self.row=num_each
        self.spiltflag = spiltflag
    def concatx(self):
        """
        连接多个数据
        暂且需要有多少数据写多少数据
        """

        data=np.zeros((self.num_train,self.sample_lenth))
        for i,data_this_doc in enumerate(self.data_doc):
            data[0+i*self.row:(i+1)*self.row]=sampling(data_this_doc,self.row,self.sample_lenth,spiltflag = self.spiltflag,spilt_ratio=0.5)
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
    根据特定的数据集构建
    '''

    test_data_name = train_data_name

    if test_rate == 0:
        testingset = readdata(test_data_name,
                             ft=len(train_data_name),
                             num_each=num_each,
                             sample_lenth=sample_lenth)

        x_test, y_test = testingset.output()
        io.savemat(data_name, {'x_test': x_test, 'y_test': y_test,})
        return x_test, y_test
    else:
        trainingset=readdata(train_data_name,
                             ft=len(train_data_name),
                             num_each=num_each,
                             sample_lenth=sample_lenth,
                             spiltflag='train')
        testingset=readdata(test_data_name,
                             ft=len(train_data_name),
                             num_each=int(num_each*test_rate),
                             sample_lenth=sample_lenth,
                             spiltflag='test')
        
        x_train, y_train = trainingset.output()
        x_test, y_test = testingset.output()
        io.savemat(data_name, {'x_train': x_train, 'y_train': y_train, 'x_test': x_test, 'y_test': y_test, })
        return x_train, y_train, x_test, y_test


if __name__ == "__main__":
    '''
    SBDS 10 classification under various load
    '''
    SBDS_root = 'D:\学习\数据集\自家试验台数据'
    
    SBDS_0K_10 = [SBDS_root+'//002//4.xlsx',
                      SBDS_root+'//010/4.xlsx',
                      SBDS_root+'//029/4.xlsx',
                      SBDS_root+'//053/4.xlsx',
                      SBDS_root+'//014/4.xlsx',
                      SBDS_root+'//037/4.xlsx',
                      SBDS_root+'//061/4.xlsx',
                      SBDS_root+'//006/4.xlsx',
                      SBDS_root+'//033/4.xlsx',
                      SBDS_root+'//057/4.xlsx'
                      ]
    SBDS_1K_10 = [SBDS_root+'//003/4.xlsx',
                      SBDS_root+'//011/4.xlsx',
                      SBDS_root+'//030/4.xlsx',
                      SBDS_root+'//054/4.xlsx',
                      SBDS_root+'//015/4.xlsx',
                      SBDS_root+'//038/4.xlsx',
                      SBDS_root+'//062/4.xlsx',
                      SBDS_root+'//007/4.xlsx',
                      SBDS_root+'//034/4.xlsx',
                      SBDS_root+'//058/4.xlsx'
                      ]
    SBDS_2K_10 = [SBDS_root+'//004/4.xlsx',
                      SBDS_root+'//012/4.xlsx',
                      SBDS_root+'//031/4.xlsx',
                      SBDS_root+'//055/4.xlsx',
                      SBDS_root+'//016/4.xlsx',
                      SBDS_root+'//039/4.xlsx',
                      SBDS_root+'//063/4.xlsx',
                      SBDS_root+'//008/4.xlsx',
                      SBDS_root+'//035/4.xlsx',
                      SBDS_root+'//059/4.xlsx'
                      ]
    SBDS_3K_10 = [SBDS_root+'//005/4.xlsx',
                      SBDS_root+'//013/4.xlsx',
                      SBDS_root+'//032/4.xlsx',
                      SBDS_root+'//056/4.xlsx',
                      SBDS_root+'//017/4.xlsx',
                      SBDS_root+'//040/4.xlsx',
                      SBDS_root+'//064/4.xlsx',
                      SBDS_root+'//009/4.xlsx',
                      SBDS_root+'//036/4.xlsx',
                      SBDS_root+'//060/4.xlsx'
                      ]
    '''
    CWRU 10 classification under various load
    '''
    CWRU_root = 'D:\学习\数据集\凯斯西储故障全部数据'
    
    CWRU_0hp_10 = [CWRU_root+'\\97.mat',
            CWRU_root+'\\105.mat',
            CWRU_root+'\\118.mat',
            CWRU_root+'\\130.mat',
            CWRU_root+'\\169.mat',
            CWRU_root+'\\185.mat',
            CWRU_root+'\\197.mat',
            CWRU_root+'\\209.mat',
            CWRU_root+'\\222.mat',
            CWRU_root+'\\234.mat']
    CWRU_1hp_10=[CWRU_root+'\\98.mat',
            CWRU_root+'\\106.mat',
            CWRU_root+'\\119.mat',
            CWRU_root+'\\131.mat',
            CWRU_root+'\\170.mat',
            CWRU_root+'\\186.mat',
            CWRU_root+'\\198.mat',
            CWRU_root+'\\210.mat',
            CWRU_root+'\\223.mat',
            CWRU_root+'\\235.mat']
    CWRU_2hp_10=[CWRU_root+'\\99.mat',
            CWRU_root+'\\107.mat',
            CWRU_root+'\\120.mat',
            CWRU_root+'\\132.mat',
            CWRU_root+'\\171.mat',
            CWRU_root+'\\187.mat',
            CWRU_root+'\\199.mat',
            CWRU_root+'\\211.mat',
            CWRU_root+'\\224.mat',
            CWRU_root+'\\236.mat']
    CWRU_3hp_10=[CWRU_root+'\\100.mat',
            CWRU_root+'\\108.mat',
            CWRU_root+'\\121.mat',
            CWRU_root+'\\133.mat',
            CWRU_root+'\\172.mat',
            CWRU_root+'\\188.mat',
            CWRU_root+'\\200.mat',
            CWRU_root+'\\212.mat',
            CWRU_root+'\\225.mat',
            CWRU_root+'\\237.mat']


    '''
    LB 4 classification under different load
    '''    
    
    LB_0t_4=['D:/学习/数据集/老师给的/列车轮对/liki_50k_50hz/0t/normal.txt',
             'D:/学习/数据集/老师给的/列车轮对/liki_50k_50hz/0t/inner.txt',
             'D:/学习/数据集/老师给的/列车轮对/liki_50k_50hz/0t/ball.txt',
             'D:/学习/数据集/老师给的/列车轮对/liki_50k_50hz/0t/outer.txt']
    LB_1t_4=['D:/学习/数据集/老师给的/列车轮对/liki_50k_50hz/1t/normal.txt',
             'D:/学习/数据集/老师给的/列车轮对/liki_50k_50hz/1t/inner.txt',
             'D:/学习/数据集/老师给的/列车轮对/liki_50k_50hz/1t/ball.txt',
             'D:/学习/数据集/老师给的/列车轮对/liki_50k_50hz/1t/outer.txt']
    LB_2t_4=['D:/学习/数据集/老师给的/列车轮对/liki_50k_50hz/2t/normal.txt',
             'D:/学习/数据集/老师给的/列车轮对/liki_50k_50hz/2t/inner.txt',
             'D:/学习/数据集/老师给的/列车轮对/liki_50k_50hz/2t/ball.txt',
             'D:/学习/数据集/老师给的/列车轮对/liki_50k_50hz/2t/outer.txt']
    LB_3t_4=['D:/学习/数据集/老师给的/列车轮对/liki_50k_50hz/3t/normal.txt',
             'D:/学习/数据集/老师给的/列车轮对/liki_50k_50hz/3t/inner.txt',
             'D:/学习/数据集/老师给的/列车轮对/liki_50k_50hz/3t/ball.txt',
             'D:/学习/数据集/老师给的/列车轮对/liki_50k_50hz/3t/outer.txt']
    
    LB_0t_3=[
             'D:/学习/数据集/老师给的/列车轮对/liki_50k_50hz/0t/inner.txt',
             'D:/学习/数据集/老师给的/列车轮对/liki_50k_50hz/0t/ball.txt',
             'D:/学习/数据集/老师给的/列车轮对/liki_50k_50hz/0t/outer.txt']
    LB_1t_3=[
             'D:/学习/数据集/老师给的/列车轮对/liki_50k_50hz/1t/inner.txt',
             'D:/学习/数据集/老师给的/列车轮对/liki_50k_50hz/1t/ball.txt',
             'D:/学习/数据集/老师给的/列车轮对/liki_50k_50hz/1t/outer.txt']
    LB_2t_3=[
             'D:/学习/数据集/老师给的/列车轮对/liki_50k_50hz/2t/inner.txt',
             'D:/学习/数据集/老师给的/列车轮对/liki_50k_50hz/2t/ball.txt',
             'D:/学习/数据集/老师给的/列车轮对/liki_50k_50hz/2t/outer.txt']
    LB_3t_3=[
             'D:/学习/数据集/老师给的/列车轮对/liki_50k_50hz/3t/inner.txt',
             'D:/学习/数据集/老师给的/列车轮对/liki_50k_50hz/3t/ball.txt',
             'D:/学习/数据集/老师给的/列车轮对/liki_50k_50hz/3t/outer.txt']
    
    LB_10hz_4=['D:/学习/数据集/老师给的/列车轮对/liki_50k_3t/10hz/normal.txt',
             'D:/学习/数据集/老师给的/列车轮对/liki_50k_3t/10hz/inner.txt',
             'D:/学习/数据集/老师给的/列车轮对/liki_50k_3t/10hz/ball.txt',
             'D:/学习/数据集/老师给的/列车轮对/liki_50k_3t/10hz/outer.txt']
    
    LB_20hz_4=['D:/学习/数据集/老师给的/列车轮对/liki_50k_3t/20hz/normal.txt',
             'D:/学习/数据集/老师给的/列车轮对/liki_50k_3t/20hz/inner.txt',
             'D:/学习/数据集/老师给的/列车轮对/liki_50k_3t/20hz/ball.txt',
             'D:/学习/数据集/老师给的/列车轮对/liki_50k_3t/20hz/outer.txt']
    
    LB_40hz_4=['D:/学习/数据集/老师给的/列车轮对/liki_50k_3t/40hz/normal.txt',
             'D:/学习/数据集/老师给的/列车轮对/liki_50k_3t/40hz/inner.txt',
             'D:/学习/数据集/老师给的/列车轮对/liki_50k_3t/40hz/ball.txt',
             'D:/学习/数据集/老师给的/列车轮对/liki_50k_3t/40hz/outer.txt']
    
    LB_50hz_4=['D:/学习/数据集/老师给的/列车轮对/liki_50k_3t/50hz/normal.txt',
             'D:/学习/数据集/老师给的/列车轮对/liki_50k_3t/50hz/inner.txt',
             'D:/学习/数据集/老师给的/列车轮对/liki_50k_3t/50hz/ball.txt',
             'D:/学习/数据集/老师给的/列车轮对/liki_50k_3t/50hz/outer.txt']
