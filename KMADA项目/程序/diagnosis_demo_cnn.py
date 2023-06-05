#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul  9 09:13:34 2019

@author: liqi
"""
# from ShuffleNetV2 import ShuffleNetV2 as sfmodel
# 导入包
from readdata import dataset
from cnn_model import CNNmodel
from os.path import splitext
import pandas as pd
import scipy
import os
import time
import torch
from torch import nn, optim
from torch.utils.data import DataLoader,TensorDataset
import numpy as np
import utils


# 基本诊断框架
class Diagnosis():
    """
    diagnosis model
    """
    def __init__(self, n_class=10, lr=0.001, batch_size=64, gpu=True, save_dir='save_dir/', model_name='default'):
        """
        接收参数后，在这个方法初始化模型
        Args:
            n_class:故障类别数
            lr:学习率
            batch_size:批量大小
            gpu:是否采用GPU
            save_dir:存储目录
            model_name:模型名字
        """

        print('diagnosis begin')

        self.net = CNNmodel(input_size=32, class_num=n_class)
        self.gpu = gpu
        self.save_dir = save_dir
        
        self.model_name = model_name
        
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
        if self.gpu:
            self.net.cuda()
        
        self.lr = lr
        self.optimizer = optim.Adam(self.net.parameters(), lr=self.lr)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=5, gamma=self.lr)
        self.loss_function = nn.CrossEntropyLoss()

        self.batch_size = batch_size

        self.train_hist = {}
        self.train_hist['loss'] = []
        self.train_hist['acc'] = []
        self.train_hist['testloss'] = []
        self.train_hist['testacc'] = []

    def caculate_acc(self, prediction, y_):
        """
        计算acc
        Args：
            prediction：模型预测值
            y_：真实标签

        """
        correct = torch.zeros(1).squeeze().cuda()
        total = torch.zeros(1).squeeze().cuda()
        correct += (prediction == y_).sum().float()
        total += len(y_)
        acc = (correct/total).cpu().detach().data.numpy()
        return acc
        
    def fit(self, x_train, y_train, x_test=None, y_test=None, epoches=1):
        """ fit方法（重要的函数需要详细描述功能）

        Args:
            x_train: 训练集数据
            y_train: 训练集标签
            x_test:测试数据
            y_test:测试标签

        Return：
            训练log ，后续会有专门的log方法，该demo先直接用字典记录

        Raises(可选)：
            写error的地方

        """
        print('training start!!')
        
        if x_test.numpy().any()==None:  #是否测试模式，历史遗留，可以忽略
            test_flag = None
        else:
            test_flag = True
            
        torch_dataset = TensorDataset(x_train, y_train)  # dataset转换成pytorch的格式
        loader =DataLoader(
            dataset=torch_dataset,     
            batch_size=self.batch_size,      
            shuffle=True,               
            num_workers=2,              
            )
        
        whole_time = time.time()
        
        for epoch in range(epoches):  # 迭代开始，1个循环为1个epoch
            print('epoch = ', epoch)
            loss_epoch = []
            acc_epoch = []
            epoch_start_time = time.time()

            for iter, (x_, y_) in enumerate(loader):  # 1个循环为 1个 batch_size
                if self.gpu:
                    x_, y_ = x_.cuda(), y_.cuda()
                
                self.optimizer.zero_grad()  # 初始化梯度
                y_pre = self.net(x_)  # 前向传播
                loss = self.loss_function(y_pre, torch.max(y_, 1)[1])  # 计算损失
                prediction = torch.argmax(y_pre, 1)  # 计算预测标签
                y_ = torch.argmax(y_, 1)   # 实际标签转化

                loss_epoch.append(loss.item())  # 保存损失
                loss.backward()  # 反向传播
                self.optimizer.step()  # 更新参数
                
                acc=self.caculate_acc(prediction, y_)  # 计算准确率
                
                acc_epoch.append(acc)  # 记录准确率
                
            epoch_end_time = time.time()  # 记录时间
            
            loss = np.mean(loss_epoch)  # 计算平均损失，作为1个epoch中的损失
            acc = np.mean(acc_epoch)  # 计算平均准确率
            epoch_time = epoch_end_time-epoch_start_time
            self.train_hist['loss'].append(loss)  # 记录该epoch的损失
            self.train_hist['acc'].append(acc)
            if test_flag:
                self.evaluation(x_test=x_test, y_test=y_test, train_step=True)  # 是否为验证模式

            if epoch % 5 == 0:  # 每5个epoch 打印
                print("Epoch: [%2d] Epoch_time:  [%8f] \n loss: %.8f, acc:%.8f" %
                              ((epoch + 1), epoch_time, loss, acc))

        total_time = epoch_end_time-whole_time
        print('Total time: %2d' % total_time)  # 记录最终结果
        print('best acc: %4f' % (np.max(self.train_hist['acc'])))
        print('best testacc: %4f' % (np.max(self.train_hist['testacc'])))

        self.save_his()  # 保存日志 ADIG模型中 写好了logger 写日志更方便
        self.save_prediction(x_test)  # 保存测试集结果
        print('=========训练完成=========')
        
        return self.train_hist

    def evaluation(self, x_test, y_test, train_step=False):
        """ 评估方法用于验证或者测试

        Args:
            x_test:测试数据
            y_test:测试标签

        Return：
            训练log ，后续会有专门的log方法，该demo先直接用字典记录

        Raises(可选)：
            写error的地方

        """
        print('evaluation')
        
        self.net.eval()
        with torch.no_grad(): 
            if self.gpu:
                x_test = x_test.cuda()
                y_test = y_test.cuda()

            y_test_ori = torch.argmax(y_test, 1)
            y_test_pre = self.net(x_test)
            test_loss = self.loss_function(y_test_pre, y_test_ori)
            y_tset_pre = torch.argmax(y_test_pre, 1)
            acc = self.caculate_acc(y_tset_pre,y_test_ori)
        print("***\ntest result \n loss: %.8f, acc:%.4f\n***" %
              (test_loss.item(),
               acc))
        
        if train_step:
            self.train_hist['testloss'].append(test_loss.item())
            self.train_hist['testacc'].append(acc)
            
            if acc >= np.max(self.train_hist['testacc']):
                self.save()
                print(' a better model is saved')

        self.net.train()
        return test_loss.item(), acc
    
    def save(self):
        """
        save model and its parameters
        """
        torch.save(self.net.state_dict(), self.save_dir+self.model_name+'.pkl')
        
    def save_prediction(self, data, data_name='test'):
        """
        save prediction
        Args:
            data：测试数据
            data_name：保存记录测试结果的文件名称
        """

        data = data.cuda()
        final_pre = self.net(data)
        
        final_pre = torch.argmax(final_pre, 1).cpu().detach().data.numpy()
        dic = {}
        dic['prediction'] = []
        dic['prediction'] = final_pre
        prediction = pd.DataFrame(dic)
        prediction.to_csv(self.save_dir+data_name+'prediction.csv')
        
    def save_his(self):
        """
        save history data
        """

        data_df = pd.DataFrame(self.train_hist)
        data_df.to_csv(self.save_dir+'history.csv')

    def load(self):
        """
        load model
        """
        self.net.load_state_dict(torch.load(self.save_dir + 'net_parameters.pkl'))


def data_reader(datadir, gpu=True):
    """
    read data from mat or other file after readdata.py
    demo只展示了mat的读取
    Args:
        datadir: 加载数据文件名称
    """
    datatype = splitext(datadir)[1]
    if datatype == '.mat':
        print('datatype is mat')

        data = scipy.io.loadmat(datadir)
        
        x_train = data['x_train']
        x_test = data['x_test']
        y_train = data['y_train']
        y_test = data['y_test']
    if datatype == '':
        pass

    x_train = torch.from_numpy(x_train)
    y_train = torch.from_numpy(y_train)
    x_test = torch.from_numpy(x_test)
    y_test = torch.from_numpy(y_test)
    return x_train, y_train, x_test, y_test


# 下面是主程序，使用`if __name__ == '__main__':`
# 的原因是避免被作为方法导入时执行下面的操作
# 精度较高的原因是CWRU的频谱特征比较明显，在数据预处理后的数据比较好分类
if __name__ == '__main__':
    
    diag = Diagnosis(n_class=10, lr=0.001, batch_size=64)  # 测试模型
    # 在下面修改加载的数据集路径
    sdatadir = 'D:/实验室数据集/datasets/CWRU_0hp_10.mat'  # be careful to capital and small letter
    tdatadir = 'D:/实验室数据集/datasets/CWRU_3hp_10.mat'
    xs_train, ys_train, xs_test, ys_test = data_reader(sdatadir)
    xt_train, yt_train, xt_test, yt_test = data_reader(tdatadir)
    diag.fit(xs_train, ys_train, xt_test, yt_test, epoches=10)

    print('源域数据集验证结果：')
    diag.evaluation(xs_test, ys_test)  # 源域数据集验证
    print('目标域数据集验证结果：')
    diag.evaluation(xt_test, yt_test)  # 目标域数据集验证
    
#    diag.load() 
    
#    diag.save_prediction(diag.x_test)

#    utils.print_network(diag.net)