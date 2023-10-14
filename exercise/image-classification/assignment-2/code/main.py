# Author: LZS
# CreateTime: 2023/10/14  20:11
# FileName: main
# Python Script

# 构建用于60×60SAR图像分类的全卷积网络

import numpy as np
import scipy.io
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import torch
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
from torch import optim
from scipy.io import loadmat
from plot_func import plot_curve, plot_image, plot_samples
from MyNet import ConvNet
from load_data import load_data
from datetime import datetime
from model_func import model_func

if __name__ == "__main__":
    # 加载数据
    train, test = load_data('mstar_data/60_60.mat', 64)

    # 显示各个类型的样本
    # plot_samples(train, figure_num=1)

    device = torch.device('cuda')  # 使用gpu训练

    model_func(train, test, device=device)
