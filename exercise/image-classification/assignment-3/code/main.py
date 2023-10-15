# Author: LZS
# CreateTime: 2023/10/15  14:53
# FileName: main
# Python Script

import sys
from datetime import datetime
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
from matplotlib import pyplot as plt
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from CNN_Net import CNN
from CNN_Net import load_model, model_CNN
from load_data import load_data
from plot_func import plot_samples, plot_curve

if __name__ == '__main__':
    train, test = load_data(minibatch_size=64)

    # images, labels = next(iter(train))
    # print(images.shape, labels.shape)  # torch.Size([64, 1, 28, 28]) torch.Size([64])

    # plot_samples(train, figure_num=1)  # 显示各个类型的样本

    device = torch.device('cuda')

    # model_path = './model/MNIST_CNN_2023-10-15_17-53.pth'
    # model = load_model(model_path, device)  # 加载模型

    model_CNN(train, test, learning_rate=0.002, epoch_num=10, device=device)
