# Author: LZS
# CreateTime: 2023/10/15  16:01
# FileName: load_data
# Python Script

import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, TensorDataset


def load_data(minibatch_size=64):
    # MNIST包含70,000张手写数字图像
    # 60,000张用于培训，10,000张用于测试。
    # 图像是灰度的，28x28像素的，并且居中的，以减少预处理和加快运行。
    transform = transforms.Compose([transforms.ToTensor(),  # 转成Tensor类型
                                    transforms.Normalize(0.1307, 0.3081)])  # 归一化 MNIST为单通道图像
    # 0.1307和0.3081是MNIST数据集的全局平均值和标准偏差
    # 导入数据
    train_set = torchvision.datasets.MNIST(root='./data',  # 数据集存放目录
                                           train=True,  # 表示是数据集中的训练集
                                           download=False,  # 第一次运行时为True，下载数据集，下载完成后改为False
                                           transform=transform)  # 预处理过程
    # 加载训练集，实际过程需要分批次（batch）训练
    train_loader = torch.utils.data.DataLoader(train_set,  # 导入的训练集
                                               batch_size=minibatch_size,  # 每批训练的样本数
                                               shuffle=True,  # 是否打乱训练集
                                               num_workers=0)  # 使用线程数，在windows下设置为0
    # 导入10000张测试图片
    test_set = torchvision.datasets.MNIST(root='./data',
                                          train=False,  # 表示是数据集中的测试集
                                          download=False,  # 第一次运行时为True，下载数据集，下载完成后改为False
                                          transform=transform)
    # 加载测试集
    test_loader = torch.utils.data.DataLoader(test_set,
                                              batch_size=10000,  # 每批用于验证的样本数
                                              shuffle=False, num_workers=0)  # 使用线程数，在windows下设置为0

    # print(len(train_set))  # 60000
    # print(len(test_set))  # 10000
    return train_loader, test_loader
