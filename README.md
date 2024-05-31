# DATA620004_HW2_resnet18finetuning

复旦大学：神经网络和深度学习  
作业2 任务1：微调在 ImageNet 上预训练的卷积神经网络实现鸟类识别

## 项目描述
本任务通过微调预训练的卷积神经网络（如 ResNet-18），实现对鸟类的识别。主要步骤包括：

1. **修改网络架构**：将 CNN 如 ResNet-18 的输出层大小设置为 200，以适应 CUB-200-2011 数据集中的类别数量。
2. **参数初始化**：使用在 ImageNet 上预训练得到的网络参数初始化除输出层外的所有层。
3. **训练新的输出层**：在 CUB-200-2011 数据集上从零开始训练新的输出层，同时对其余层进行微调。
4. **超参数优化**：调整训练步数、学习率等参数，观察不同配置对模型性能的影响。
5. **性能比较**：与从随机初始化开始训练的网络进行对比，分析预训练参数的效果。

## 安装依赖

安装必需的 Python 库：

```bash
pip install torch tensorboard torchvision
```

## 训练指南

### 微调预训练网络

1. 打开 `train_fine_tunever.py`，设置数据集路径及参数搜索范围。
2. 在 `train_fine_tune_model()` 函数中，根据需求修改训练学习率和 epoch 数。
3. 执行以下命令以启动训练：

```bash
python train_fine_tunever.py
```

### 从随机初始化微调

1. 修改 `train_from_scratch.py` 文件以调整必要的设置。
2. 执行以下命令以启动训练：

```bash
python train_from_scratch.py
```

## 测试模型

1. 在 `codes.test_model` 中设置数据集路径和权重保存位置。
2. 运行以下命令进行测试：

```bash
python test_model.py
```
