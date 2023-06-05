# 基于相似性度量领域自适应的轴承故障诊断方法（Similarity Measurement Domain Adaptation，SMDA）

## 1. 项目学习须知
### 1.1上传项目注意事项
- 提出的新方法程序介绍（程序备注）最好多一些，同时说明文档中标明出处；  
- 每个.py程序文件都可以独立运行最好可观察调试结果（根据情况而定）；  
- 包含可直接加载数据集，如果数据集不大可以直接放入项目文件中。
### 1.2 需要读的参考文献
- 小论文 《基于相似性度量迁移学习的轴承故障诊断》或大论文第三章
- 相似性度量模块参考论文 《Predicting with High Correlation Features》

### 1.3使用的数据集
- 苏州大学数据集SBDS  
  - 数据集四种工况文件名：SBDS_0K_10.mat、SBDS_1K_10.mat、SBDS_2K_10.mat、SBDS_3K_10.mat  
  - 训练集尺寸：size = [4000, 1, 32, 32]  
  - 测试集尺寸：size = [2000, 1, 32, 32]  
- 凯斯西储大学数据集CWRU
  - 数据集四种工况文件名：CWRU_0hp_10.mat、CWRU_1hp_10.mat、CWRU_2hp_10.mat、CWRU_3hp_10.mat
  - 训练集尺寸：size = [4000, 1, 32, 32]  
  - 测试集尺寸：size = [2000, 1, 32, 32]
## 2.项目概述
### 2.1 解决的问题
- 在跨域情况下如何使模型提取到有效的可迁移特征；
- 高相关性特征能更正确地表示对应的故障类型，工况变化时低相关性特征往往更容易产生偏差，如何增加高相关性特征的贡献度。

### 2.2 创新点
- 将相似性度量的想法加入领域自适应； 
- 通过相关对齐损计算故障特征之间的相关性，最小化源域和目标域特征之间的分布差异； 
- 同时最大化输入特征与中心特征的相似性，提高故障特征聚类的准确性。
### 2.3 用的核心技术
- CNN；  
- 领域自适应；  
- 相似性度量。  

### 2.4 模型图
![节点](1.png)

## 3. 程序包含的.py文件
### 3.1 程序所包含的具体函数介绍
- #### readdata.py：**数据读取预处理**
    - `shuru`：原始数据读取
    - `norm`：可以采用sklearn中的库，数据标准化
    - `sampling`：从序列中采样样本
    - `readdata`：实现读取-数据预处理-采样的工作流，构建一个样本
    - `dataset`：在readdata的基础上重复执行readdate的流，来构建数据集
    - `main`函数：input 数据集地址（本项目中没用到，datasets中已有处理好可以直接用于诊断的数据）
- #### diagnosis_demo_cnn.py： **诊断程序，运行此文件完成故障诊断**
    - `Diagnosis`：定义一个诊断的类
        - `caculate_acc`：计算精度（评价指标）
        - `fit`：定义一个训练的方法，包含主要的训练步骤
        - `evaluation`：用于测试
        - `save`：储存模型和参数
        - `save_prediction`：储存预测精度，保存为csv文件
        - `save_his`：储存训练记录（loss和训练精度），保存为csv文件
        - `load`：加载模型
    - `data_reader`：读取.mat文件的数据
    - `load_data`：加载数据
    - `main`函数：主程序设置迁移任务和各个参数
- #### cnn_model.py： **CNN模型程序**
    - `conv1x1`：自定义1X1卷积模块
    - `CNNmodel`：利用框架构架模型，明确输入输出的size
        - `mid_rep`：输出中间某一层（用于画中间层的可视化）
        - `forward`：前向传播
        - `predict`：预测
        - `adapt_loss`：计算源域和目标域之间的域自适应损失
    - `main`函数：主程序随机生成input 测试模型骨架
- #### correlation.py： **相似性度量模块**
    - 数学计算，参考论文Predicting with High Correlation Features
- #### utils.py： **辅助工具**
    - `make_cuda`：调用cuda
    - `loop_iterable`：返回迭代对象中的每个元素
    - `print_network`：打印网络结构
    - `initialize_weights`：初始化网络参数， xavier 还是kaiming


###**3.2 模型说明：cnn_model.py**
- #### 包含的层结构
    - Conv2d 3×3
    - LeakyReLU
    - AvgPool平均池化
    - 全连接层 = `conv1×1`
- #### forward传入参数
    - `source`：源域训练数据
    - `target`：目标域训练数据
    - `ret_hid`： 是否输出hid
- #### forward返回参数
    - `source_clf`：源域预测标签
    - `target_clf`：目标域预测标签
    - `transfer_loss`：coral损失
    - `hid`：第一层卷积后的特征

###**3.3 诊断主程序说明：diagnosis_demo_cnn.py**
- #### 可调整的参数
    - lr=0.0001
    - batch_size=64
    - loss各项的权重系数
- #### 损失函数
    - `clf_loss`：交叉熵损失函数，用于计算模型预测输出与真实标签的损失[ys_train_pre, torch.max(ys_train)]
    - `regularization_loss`：correlation_reg计算输入特征与中心特征之间的相似性度量[(hid_repr, yt_train_pre)]
    - `transfer_loss`：CORAL损失
