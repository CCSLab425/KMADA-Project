#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul  9 09:13:34 2019

@author: liqi
"""

# 导入一些需要的模块
from readdata import dataset  # 数据读取
from cnn_model import CNNmodel  # CNN模型
from os.path import splitext  # 读取文件类型
import pandas as pd
import scipy  # 科学处理库
import os  # os操作库
import time  # 时间库
import torch  # 导入pytorch框架
from torch import nn, optim  # 神经网络，优化
from torch.utils.data import DataLoader,TensorDataset  # 数据读取模块
import numpy as np
import utils  # 一些需要的功能
from correlation import correlation_reg  # 相似性度量模块
import scipy.io as io


# 首先定义一个诊断的类
class Diagnosis():
    """
    diagnosis model
    """
    def __init__(self, n_class=10, lr=0.001, batch_size=64,
                 gpu=True, save_dir='SBDSsave_dir/SMDA', tl_task='0to1', model_name='default', target_name='default_source'):
        """
        初始化函数
        包括分类数、学习率、批处理大小、GPU状态、存储路径、任务名称、模型名称、目标域名称
        后两个可能没用到，定义了也不影响
        """
    
        print('diagnosis begin')
        self.net = CNNmodel(input_size=32,class_num=n_class)
        self.gpu = gpu
        self.save_dir = save_dir
        self.target_name = target_name
        self.tl_task = tl_task
        self.model_name = model_name

        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
        if self.gpu:
            self.net.cuda()
        
        self.lr = lr
        self.optimizer = optim.Adam(self.net.parameters(), lr=self.lr)  # 使用Adam优化器优化
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer,step_size=5,gamma = self.lr)  # 调度程序
        self.loss_function = nn.CrossEntropyLoss()  # 用交叉熵计算损失
        self.batch_size = batch_size

        self.train_hist = {}  # 定义一个字典储存训练记录
        self.train_hist['loss'] = []  # 总损失
        self.train_hist['clf_loss'] = []  # 分类损失 
        self.train_hist['regularization_loss'] = []  # 相似性度量损失
        self.train_hist['transfer_loss'] = []  # CORAL损失
        self.train_hist['acc'] = []  # 训练精度
        self.train_hist['testloss'] = []  # 测试损失
        self.train_hist['testacc'] = []  # 测试精度

    def caculate_acc(self, prediction, y_):
        """
        计算acc
        """
        correct = torch.zeros(1).squeeze().cuda()
        total = torch.zeros(1).squeeze().cuda()
        correct += (prediction == y_).sum().float()
        total+=len(y_) 
        acc=(correct/total).cpu().detach().data.numpy()
        return acc
        
    def fit(self, s_train_loader, t_train_loader, xt_test, yt_test, epoches=1):
        """
        fit方法，主要的训练步骤
        源域训练集 s_train_loader，xs_train 源域训练数据，ys_train 源域训练标签
        目标域训练集 t_train_loader，xt_train 目标域训练数据，yt_train 目标域训练标签
        目标域测试集 xt_test,yt_test
        """
        print('training start!!')
        
        whole_time=time.time()
        print('training start!!')
        
        for epoch in range(epoches):  # 训练循环开始
            loss_epoch = []
            clf_loss_epoch = []
            regularization_loss_epoch = []
            transfer_loss_epoch = []
            acc_epoch = []
            
            epoch_start_time = time.time()
            
            for iter, ((xs_train, ys_train), (xt_train,yt_train)) in enumerate(zip(s_train_loader,t_train_loader)):  # 一次迭代一个epoch
                self.optimizer.zero_grad()
                if self.gpu:  # 调用cuda
                    xs_train, xt_train = xs_train.cuda(), xt_train.cuda()
                    xt_test, yt_test = xt_test.cuda(),yt_test.cuda()
                    ys_train = ys_train.cuda()
                    
                # ============================
                # 前向传播-计算各个loss
                ys_train_pre,yt_train_pre,transfer_loss,hid_repr = self.net(xs_train,xt_train,ret_hid=True)
                clf_loss = self.loss_function(ys_train_pre, torch.max(ys_train, 1)[1])
                prediction = torch.argmax(ys_train_pre, 1)
                ys_train=torch.argmax(ys_train,1)
                
                regularization_loss = 0
                regularization_loss = correlation_reg(hid_repr, yt_train_pre)
                # 计算总的loss函数
                loss = clf_loss + 0.001 * regularization_loss + 10 * transfer_loss
                # ============================

                # ============================
                # 反向传播-到模型收敛后训练结束
                loss_epoch.append(loss.item())
                loss.backward()  # 返回参数
                clf_loss_epoch.append(clf_loss.item())
                regularization_loss_epoch.append(regularization_loss.item())
                transfer_loss_epoch.append(transfer_loss.item())
                self.optimizer.step()  # 优化 反向参数 梯度下降
                # ============================
                
                acc = self.caculate_acc(prediction,ys_train)
                acc_epoch.append(acc)
            print('training start!!')
            epoch_end_time = time.time()
           
            loss = np.mean(loss_epoch)
            clf_loss = np.mean(clf_loss_epoch)
            regularization_loss = np.mean(regularization_loss_epoch)
            transfer_loss = np.mean(transfer_loss_epoch)
            acc = np.mean(acc_epoch)
            epoch_time = epoch_end_time-epoch_start_time

            self.train_hist['loss'].append(loss)
            self.train_hist['clf_loss'].append(clf_loss)
            self.train_hist['regularization_loss'].append(regularization_loss)
            self.train_hist['transfer_loss'].append(transfer_loss)
            self.train_hist['acc'].append(acc)
            self.evaluation(xt_test,yt_test, train_step=True)

            if epoch%1==0:
                print("Epoch: [%2d] Epoch_time:  [%8f] \n loss: %.8f, acc:%.8f" %
                              ((epoch + 1),epoch_time,loss,acc))

        total_time=epoch_end_time-whole_time
        print('Total time: %2d'%total_time)
        print('best acc: %4f'%(np.max(self.train_hist['acc'])))
        print('best testacc: %4f'%(np.max(self.train_hist['testacc'])))

        self.save_his()
        self.save_prediction(xt_test)
        print('=========训练完成=========')
        
#        return self.train_hist,loss,clf_loss,regularization_loss
        return self.train_hist
        
    def evaluation(self,xt_test,yt_test,train_step=False):
        """
        用于测试
        x_test:测试数据
        y_test:测试标签
        """
        print('evaluation')
        
        self.net.eval()  # 测试的时候一定不要忘记这一步，相应的权重更新或者drop的更新关闭
        if self.gpu:
            xt_test, yt_test = xt_test.cuda(),yt_test.cuda()
        
        with torch.no_grad():  # 关闭梯度计算
        
            yt_test_ori=torch.argmax(yt_test,1)
            yt_test_pre=self.net.predict(xt_test)
            test_loss=self.loss_function(yt_test_pre,yt_test_ori)
            yt_test_pre=torch.argmax(yt_test_pre,1)

            acc=self.caculate_acc(yt_test_pre,yt_test_ori)
              
        print("***\ntest result \n loss: %.8f, acc:%.4f\n***" %
              (test_loss.item(),acc))
        
        if train_step:
            self.train_hist['testloss'].append(test_loss.item())
            self.train_hist['testacc'].append(acc)

            if acc>=np.max(self.train_hist['testacc']) :
                self.save()

                print(' a better model is saved')

        self.net.train()
        return test_loss.item(),acc
    
    def save(self):
        """
        储存模型和参数
        """
        torch.save(self.net.state_dict(), self.save_dir+self.tl_task+'.pkl')
        
    def save_prediction(self,data,data_name='test'):
        """
        储存预测精度，保存为csv文件
        """
        data=data.cuda()
        
        final_pre=self.net.predict(data)
        
        final_pre=torch.argmax(final_pre,1).cpu().detach().data.numpy()
        dic={}
        dic['prediction']=[]
        dic['prediction']=final_pre
        prediction=pd.DataFrame(dic)
        prediction.to_csv(self.save_dir+self.tl_task+'_pre.csv')
        
    def save_his(self):
        """
        储存训练记录，保存为csv文件
        """
        data_df = pd.DataFrame(self.train_hist)
        data_df.to_csv(self.save_dir+ self.tl_task +'.csv')
        
    def load(self):
        """
        加载模型
        """
        self.net.load_state_dict(torch.load(self.save_dir +'net_parameters.pkl'))

def data_reader(datadir,gpu=False):
    """
    读取处理好的数据集，以mat文件为例
    """
    datatype=splitext(datadir)[1]
    if datatype=='.mat':

        print('datatype is mat')
        data=scipy.io.loadmat(datadir)      
        x_train=data['x_train']
        x_test=data['x_test']     
        y_train=data['y_train']     
        y_test=data['y_test']
    if datatype=='':
        pass
    
    x_train=torch.from_numpy(x_train) #转化成tensor张量
    y_train=torch.from_numpy(y_train)
    x_test=torch.from_numpy(x_test)
    y_test=torch.from_numpy(y_test)
    return x_train,y_train,x_test,y_test

def load_data(x,y):
    """
    加载数据
    """
    torch_dataset = TensorDataset(x,y)
    loader = DataLoader(
        dataset = torch_dataset,
        batch_size=64,
        shuffle=True,
        num_workers=0,
    )
    return loader


# 主程序
if __name__ == '__main__':
    
    diag=Diagnosis(n_class = 10, lr = 0.0001, batch_size = 64, gpu=True, 
                        save_dir='SBDSsave_dir/SMDA', tl_task='0to1')  #实例化诊断任务，有GPU的请让变量gpu=True
    sdatadir = 'datasets/SBDS_0K_10.mat'  # 输入源域数据路径
    tdatadir = 'datasets/SBDS_1K_10.mat'  # 输入目标域数据路径
    xs_train,ys_train,xs_test,ys_test = data_reader(sdatadir)  # 读取数据
    s_train_loader = load_data(xs_train,ys_train)
    xt_train,yt_train,xt_test,yt_test = data_reader(tdatadir)
    t_train_loader = load_data(xt_train,yt_train)  # 读取数据
    
    diag.fit(s_train_loader,t_train_loader,xt_test,yt_test,epoches = 100)  # fit模型
    diag.evaluation(xt_test,yt_test)  # 评估模型
    
    """
    下面这部分是用来输出tsne文件的
    """
    E = diag.net.cpu().mid_rep
    Fs = E(xs_test).view(2000,-1)
    Ft = E(xt_test).view(2000,-1)
    
    Fs = Fs.detach().numpy()
    Ft = Ft.detach().numpy()
  
    tl_task='0to1'        
    save_dir2 = 'SBDS_save_dir/SMDA/feature2/'
    
    if not os.path.exists(save_dir2):
        os.makedirs(save_dir2)
    
    io.savemat(save_dir2+ tl_task +'_features.mat',{'Fs': Fs,'Ft': Ft,})
   

