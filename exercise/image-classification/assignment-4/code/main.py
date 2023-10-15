# Author: LZS
# CreateTime: 2023/10/15  21:41
# FileName: main
# Python Script


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
from load_data import load_data, get_first_data
from datetime import datetime
from model_func import model_func, load_model, model_visual

train, test = load_data('mstar_data/88_88.mat', 64)

# inputs, labels = next(iter(train))
# print(inputs.shape, labels.shape)  # torch.Size([64, 1, 88, 88]) torch.Size([64])
# print(inputs.shape[2], inputs.shape[3])  # torch.Size([64, 1, 88, 88]) torch.Size([64])

# plot_samples(train, figure_num=1)  # 显示各个类型的样本

device = torch.device('cuda')  # 使用gpu训练，指定device

model = model_func(train, test, epoch_num=10, device=device)  # 训练模型

