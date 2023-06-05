"""
Created on Mon Jun 10 19:49:25 2019
@author: 李奇
"""
# 导入包
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
import numpy as np
import utils
import torch.nn.functional as F
from torchstat import stat
from ptflops import get_model_complexity_info


# 1X1卷积模块，可以参考ResNet的瓶颈层
def conv1x1(in_channels, out_channels, bias=True, groups=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=bias, groups=groups)


# 卷积模型包括特征提取器+分类器
class CNNmodel(nn.Module):
    def __init__(self, input_size=32, class_num=10):
        """
        Args:
            input_size:输入数据尺寸
            class_num:故障分类数量
        """
        super(CNNmodel, self).__init__()
        
        self.input_size = input_size
        self.class_num = class_num

        # 特征提取器
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(1, 64, 3, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 64, 3, 2, 1),
            nn.LeakyReLU(0.2),
            # 16
            
            nn.Conv2d(64, 128, 3, padding=1),            
            nn.LeakyReLU(0.2),
            nn.Conv2d(128, 128, 3, 2, 1),            
            nn.LeakyReLU(0.2),
            # 8
            
            nn.Conv2d(128, 256, 3, padding=1),
            nn.LeakyReLU(0.2),
        )

        # 分类器1
        self.clf = nn.Sequential(
            nn.Conv2d(256, 256, 3, 2, 1),
            nn.LeakyReLU(0.2),
            nn.AvgPool2d(4),  # 平均池化
            conv1x1(256, self.class_num)
                )
        # 分类器2
        self.clf2 = nn.Sequential(
            nn.Conv2d(256, 256, 3, 2, 1),
            nn.Flatten(),
            # nn.Linear(4096, 1024),
            nn.Linear(4096, 10)
                )
        utils.initialize_weights(self)

    def dim_show(self, input):
        """
        测试net函数内每一层的输出维度
        Args:
            input: 测试数据

        Returns:详细通道与尺寸

        """
        X = input
        print('特征提取器各层size：')
        for layer in self.feature_extractor:
            X = layer(X)
            print(layer.__class__.__name__, 'output shape: \t', X.shape)
        print('全连接层size：')
        for layer in self.clf:
            X = layer(X)
            print(layer.__class__.__name__, 'output shape: \t', X.shape)
            pass

    def forward(self, x, output_flag=True, clf_flag='1'):
        """
        Args：
            x:输入数据
            output_flag:分类输出
            clf_flag:分类器选择
        return：
            分类预测输出 or 提取特征
        """
        x = self.feature_extractor(x)

        if clf_flag == '1':
            c = self.clf(x)
        
        elif clf_flag == '2':
            c = self.clf2(x)
        
        if output_flag:
            c = c.view(c.size(0), c.size(1))  # [1,10,1,1]变二维
            return c
        else:
            return x


# 领域判别器
class Dmodel(nn.Module):
    def __init__(self, input_size=[256, 8, 8]):
        """
        Args:
            input_size：输入尺寸
        """
        super(Dmodel, self).__init__()
        self.input_size = input_size
        self.Dclf = nn.Sequential(
                
                nn.Conv2d(256, 128, 3, 2, 1),  # 下采样 size=4
                nn.LeakyReLU(0.2),
                nn.Conv2d(128, 64, 3, 2, 1),  # 下采样 size=2
                nn.LeakyReLU(0.2),
                nn.AvgPool2d(2),  # 平均池化，size=1
                conv1x1(64, 1)
                # nn.Flatten(),
                # nn.Linear(64*4,1)
                )

        utils.initialize_weights(self)

    def forward(self, dx):
        """
        Args:
            dx:输入数据
        return：
            域标签
        """

        dx = self.Dclf(dx)
        return dx.view(dx.size(0), dx.size(1))


#  测试模型结构是否存在问题
if __name__ == '__main__':

    model = CNNmodel(input_size=32, class_num=10)

    x = torch.randn(1, 1, 32, 32)
    output = model.forward(x, clf_flag='1')
    
    # utils.print_network(net)
    model.dim_show(x)
    print('前向传播最终输出：')
    print('forward_output.shape = ', output.shape)
    flops, params = get_model_complexity_info(model, (1, 32, 32), as_strings=True, print_per_layer_stat=True,
                                              verbose=True)
    stat(model, (1, 32, 32))
    
