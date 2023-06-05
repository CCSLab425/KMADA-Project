# 基于知识映射的对抗领域自适应方法（KMADA）说明文档
##1. 项目学习须知
###1.1上传项目注意事项
- 提出的新方法程序介绍（程序备注）最好多一些，同时说明文档中标明出处；  
- 每个.py程序文件都可以独立运行最好可观察调试结果（根据情况而定）；  
- 包含可直接加载数据集，如果数据集不大可以直接放入项目文件中。
###1.2需要读的参考文献
- 《基于对抗学习的旋转机械跨域故障诊断研究》——第三章  
- 《Generative Adversarial Nets》  
- 《Domain-Adversarial Training of Neural Networks》  
- 《Unsupervised Domain Adaptation by Backpropagation》
###1.3使用的数据集
- 苏州大学数据集SBDS  
 - 数据集四种工况文件名：SBDS 0K 10.mat、SBDS 1K 10.mat、SBDS 2K 10.mat、SBDS 3K 10.mat。  
 - 训练集尺寸：size = [4000, 1, 32, 32]  
 - 测试集尺寸：size = [2000, 1, 32, 32]  
- 凯斯西储大学数据集CWRU  
 - 数据集四种工况文件名：CWRU 0hp 10.mat、CWRU 1hp 10.mat、CWRU 2hp 10.mat、CWRU 3hp 10.mat。  
 - 训练集尺寸：size = [4000, 1, 32, 32]  
 - 测试集尺寸：size = [2000, 1, 32, 32]
##2.项目概述
###2.1 解决的问题
- 现阶段基于人工智能技术的机械故障诊断方法广受研究者的青睐，然而由于机械常常服役于变工况，从源域学习到的模型只能诊断源域信号，在目标域的模式识别任务中，往往无法取得令人满意的性能。
- 受到 GAN 能够无监督地挖掘特征的启发，针对变工况下跨域故障诊断问题，提出了一种基于知识映射的对抗领域自适应(Knowledge mapping-based adversarial domain adaptation，KMADA) 方法

###2.2 创新点
- 引入对抗域适应的方法实现跨工况故障诊断；  
- 添加源域预训练模型，将预训练的源域模型参数迁移至目标域模型，可使模型训练快速收敛，提升训练速度；  
- 引入网格学习率自搜索策略，提升模型的泛化性能。
###2.3 用的核心技术
- 对抗（GAN）训练技术；  
- 领域自适应技术；  
- 学习率自搜索技术。  
###2.4 所做的贡献
- 提出一种利用源知识识别目标域故障的 KMADA 模型，其利用迁移学习实现目标域无监督信号的跨域识别与诊断。与现有 TL 方法相比，所提方法能够将知识从目标域推广到源域，在各种工况下达到最高的精度。  
- 对跨域不变特征提取进行研究，在对抗学习的启发下，利用特征提取器提取域不变并且故障可区分的特征，其中目标特征提取器受域判别器的约束。  
- 对抗领域自适应研究，在对抗学习的启发下，利用领域判别器进行领域差异估计，不再需要将人工设计的距离测量嵌入到网络结构中，从而增强了模型的可行性和通用性。  
- 此外，利用学习速率网格搜索 (LRGS) 方案和预训练步骤，所提方法可快速达到最优平衡点，加速模型敛。
##3. KMADA项目程序文件介绍
###3.1 项目文件概述
- 诊断模型：`cnn_model`；  
- 读取数据：`readdata`；  
- KMADA诊断程序：`KMADA_diagnosis`；  
- KMADA诊断网格搜索最佳学习率：`KMADA_diagnosis_lr_ex`；   
- 短时傅里叶变换程序：`STFT`；  
- 其它函数封装程序：`uilts`。  
每个程序所包含的具体函数介绍如下。
###3.2 项目文件包含的函数或类介绍
- **`cnn_model.py`** 
 - `CNNmodel`:卷积模型包括特征提取器与分类器；  
 - `Dmodel`:判别器模型；  
 - `forward`：前向传播
- **`readdata.py`**  
 - `load_data`：原始数据读取；  
 - `norm`：可以采用sklearn中的库；  
 - `sampling`： 从序列中采样样本；  
 - `readdata`：类实现读取-数据预处理-采样的工作流，构建一个样本；  
 - `dataset`：在readdata的基础上重复执行readdate的流，来构建数据集；  
 - `main`：输入数据集地址。  
- **`KMADA_diagnosis.py`**  
 - `data_reader`:从mat文件中读取数据；  
 - `TL_diagnosis`：迁移任务诊断；  
 - `fit`：诊断函数；  
 - `evaluation`：评估函数；  
 - `save_his`：保存历史数据；
 - `main`：诊断主函数。
- **`KMADA_diagnosis_lr_ex.py`**  
 - `data_reader`:从mat文件中读取数据；  
 - `TL_diagnosis`：迁移任务诊断；  
 - `fit`：诊断函数；  
 - `evaluation`：评估函数；  
 - `save_his`：保存历史数据；  
 - `main`：诊断主函数，在for循环中寻找最佳学习率。
- **`STFT.py`**  
 - `stft`：对原始数据进行短时傅里叶变换；  
 - `stft_specgram`：绘制频谱图像并保存。
- **`uilts.py`**  
 - `make_cuda`：选择训练的cuda；  
 - `set_requires_grad`：参数梯度更新；  
 - `caculate_acc`：计算诊断准确率；  
 - `lr_changer`：学习率自衰减；  
 - `lossplot`：绘制诊断损失图像；  
 - `accplot`：绘制诊断准确率图像；  
 - `figscat`：图像拼接；  
 - `print_network`：打印网络模型参数量；  
 - `initialize_weights`：初始化模型参数。
###3.3 诊断模型说明
-  包含的层结构
    - Conv2d 3×3
    - LeakyReLU
    - AvgPool
    - conv1×1(全连接层)
- forward传入参数
    - `x`：输入数据
    - `output_flag`：分类输出
    - `clf_flag`： 分类器选择
- forward返回参数
    - `c`：故障分类预测标签
    - `x`：特征提取器提取的特征
###3.4 KMADA诊断主程序说明
- 可调整的超参数
    - 源域学习率：lr=0.001
    - 源域小批量大小：batch_size=64
    - 目标域学习率：lr=0.001
    - 目标域小批量大小：batch_size=64
    - 判别器学习率：Dlr=0.0001
    - 特征提取器学习率：Flr=0.001
- 损失函数
    - `clf_loss`：交叉熵损失函数，用于计算模型预测输出与真实标签的损失
    - `BCEWithLogitsLoss`：二元分类损失，用于判别器判别源域与目标域的域标签损失计算
