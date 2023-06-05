"""
Created on Mon Jun 10 19:49:25 2019
@author: 李奇
"""
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
import numpy as np
import utils
import torch.nn.functional as F


DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# 自定义模块1X1卷积
def conv1x1(in_channels,out_channels,bias=True,groups=1):
    return nn.Conv2d(in_channels,out_channels,kernel_size=1,stride=1,padding=0,bias=bias,groups=groups)#卷积


# 构架一个CNN模型
class CNNmodel(nn.Module):
    def __init__(self,input_size=32, class_num=10):
        super(CNNmodel, self).__init__()
        
        self.input_size = input_size
        self.class_num = class_num

        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 64, 3, padding=1)  # 输入维度1 输出维度64
            )
            
        self.feature_ex = nn.Sequential(
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 64, 3, 2, 1),
            nn.LeakyReLU(0.2),
            
            nn.Conv2d(64, 128, 3, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(128, 128, 3, 2, 1),
            nn.LeakyReLU(0.2),
               
            nn.Conv2d(128, 256, 3, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(256, 256, 3, 2, 1),
            nn.LeakyReLU(0.2))

        self.clf = nn.Sequential(
                nn.AvgPool2d(4),
                conv1x1(256,256),
                conv1x1(256,self.class_num)
                )
        utils.initialize_weights(self)  # 参数随机初始化
    
    # 输出中间某一层，可用来画中间层的可视化，诊断模型中没有用
    def mid_rep(self,x):
        x = self.conv1(x)
        x = self.feature_ex(x)
        return x
    
    # 前向传播
    def forward(self,source,target,output_flag=True,ret_hid=False):
        hid = self.conv1(source)
        print('hid.shape = ', hid.shape)
        source = self.feature_ex(hid)
        target = self.conv1(target)
        target = self.feature_ex(target)
        source_clf = self.clf(source)
        source_clf = source_clf.view(source_clf.size(0),source_clf.size(1))
        target_clf = self.clf(target)
        target_clf = target_clf.view(target_clf.size(0),target_clf.size(1))
        print('source1.shape = ', source.shape)
        print('target1.shape = ', target.shape)
        source = source.view(source.size(0), -1)
        target = target.view(target.size(0), -1)
        coral_loss = self.CORAL_loss(source, target)
        if ret_hid:
            return source_clf,target_clf,coral_loss, hid
        
        return source_clf,target_clf,coral_loss

    # 预测
    def predict(self,input):
        x = self.conv1(input)
        x = self.feature_ex(x)
        c = self.clf(x)
        c = c.view(c.size(0),c.size(1))
        return c

    # 计算源域和目标域之间的CORAL的部分
    def CORAL_loss(self, source_feature, target_feature):
        """
        计算源域和目标域之间的CORAL
        source_feature：源域特征，二维数据
        target_feature：目标域特征，二维数据
        """
        d = source_feature.size(1)
        ns, nt = source_feature.size(0), target_feature.size(0)  # 源域与目标域样本的数量
        source = source_feature.cuda(0)
        target = target_feature.cuda(0)

        # source covariance
        tmp_s = torch.ones((1, ns)).to(DEVICE) @ source
        cs = (source.t() @ source - (tmp_s.t() @ tmp_s) / ns) / (ns - 1)  # 源域特征的协方差矩阵
    
        # target covariance
        tmp_t = torch.ones((1, nt)).to(DEVICE) @ target
        ct = (target.t() @ target - (tmp_t.t() @ tmp_t) / nt) / (nt - 1)  # 目标域特征的协方差矩阵

        # Frobenius 范数，对应元素的平方和，再开方
        coral_loss = (cs - ct).pow(2).sum().sqrt()
        loss = coral_loss / (4 * d * d)
        
        return loss


# 可以运行主程序随机生成input，测试模型是否通畅   
if __name__=='__main__':
    net = CNNmodel(input_size=32, class_num=10)
    input = torch.randn(64,1,32,32)
    target = torch.randn(64,1,32,32)
    output = net.forward(input, target)
    
    utils.print_network(net.clf)
    
