import os
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from sklearn.model_selection import train_test_split

# 数据切分函数
def split_dataset(dataset, train_ratio=0.8):
    train_indices = []
    test_indices = []

    # 获取数据集的所有标签
    targets = np.array(dataset.targets)
    # 获取所有唯一的类别标签
    classes = np.unique(targets)
    
    for cls in classes:
        # 找到属于当前类别的所有样本的索引
        cls_indices = np.where(targets == cls)[0]
        
        # 按照给定比例划分训练集和测试集
        train_cls_indices, test_cls_indices = train_test_split(cls_indices, train_size=train_ratio, shuffle=True)
        
        # 将划分后的索引加入对应的列表中
        train_indices.extend(train_cls_indices)
        test_indices.extend(test_cls_indices)
    
    return train_indices, test_indices

# 加载数据函数
def load_my_data(data_dir):
    # 数据预处理和增强
    data_transforms = {
        'train': transforms.Compose([
            transforms.Resize(256),
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225])
        ]),
        'test': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225])
        ]),
    }

    dataset = datasets.ImageFolder(data_dir)
    
    # 按类别划分训练和测试集
    train_indices, test_indices = split_dataset(dataset, 0.8)

    train_dataset = Subset(dataset, train_indices)
    test_dataset = Subset(dataset, test_indices)

    # 再次按比例将训练集切分为训练集和验证集
    train_indices, vali_indices = split_dataset(train_dataset.dataset, 0.8)
    train_dataset = Subset(train_dataset.dataset, train_indices)
    vali_dataset = Subset(train_dataset.dataset, vali_indices)

    # transform图像增强 
    train_dataset.dataset.transform = data_transforms['train'] 
    vali_dataset.dataset.transform = data_transforms['test'] 
    test_dataset.dataset.transform = data_transforms['test'] 
    # 创建数据加载器 
    dataloaders = { 'train': DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4), 'vali': DataLoader(vali_dataset, batch_size=32, shuffle=True, num_workers=4), 'test': DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4) } 
    # 获取数据集大小和类别名称 
    dataset_sizes = {'train': len(train_dataset), 'vali': len(vali_dataset), 'test': len(test_dataset)} 
    class_names = dataset.classes 
    print("Classes:", class_names) 
    print("Dataset sizes:", dataset_sizes) 
    return dataloaders, dataset_sizes