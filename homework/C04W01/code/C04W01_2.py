# Author: LZS
# CreateTime: 2023/10/13  15:59
# FileName: C04W01_2
# Python Script

# 卷积神经网络实例
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset

import math
import numpy as np
import h5py
import matplotlib.pyplot as plt
import scipy
from PIL import Image
from scipy import ndimage

from cnn_utils import *

np.random.seed(1)

X_train_orig, Y_train_orig, X_test_orig, Y_test_orig, classes = load_dataset()

# print("X_train shape: " + str(X_train_orig.shape))  # (1080, 64, 64, 3)
# print("Y_train shape: " + str(Y_train_orig.shape))  # (1, 1080)
# print("X_test shape: " + str(X_test_orig.shape))  # (120, 64, 64, 3)
# print("Y_test shape: " + str(Y_test_orig.shape))  # (1, 120)

# x归一化 y转hot
mean = X_train_orig.mean()
std = X_train_orig.std()
X_train_norm = (X_train_orig - mean) / std
Y_train_norm = Y_train_orig.reshape(Y_train_orig.shape[1])
X_test_norm = (X_test_orig - mean) / std
Y_test_norm = Y_test_orig.reshape(Y_test_orig.shape[1])

# 转为 torch.Tensor 类型
X_train = torch.tensor(X_train_norm, dtype=torch.float).permute(0, 3, 1, 2)
Y_train = torch.tensor(Y_train_norm, dtype=torch.long)
X_test = torch.tensor(X_test_norm, dtype=torch.float).permute(0, 3, 1, 2)
Y_test = torch.tensor(Y_test_norm, dtype=torch.long)

# print("X_train shape: " + str(X_train.shape))  # torch.Size([1080, 3, 64, 64])
# print("Y_train shape: " + str(Y_train.shape))  # torch.Size([1080])
# print("X_test shape: " + str(X_test.shape))  # torch.Size([120, 3, 64, 64])
# print("Y_test shape: " + str(Y_test.shape))  # torch.Size([120])

# if torch.cuda.is_available():
#     device = torch.device("cuda")
# else:
#     device = torch.device("cpu")

device = torch.device("cuda")


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.model_conv = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=8, kernel_size=4, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=8, stride=8, padding=4),
            nn.Conv2d(in_channels=8, out_channels=16, kernel_size=2, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=4, stride=4, padding=2),
        )
        self.model_linear = nn.Sequential(
            nn.Flatten(),
            nn.Linear(144, 6)  # 全连接层
        )

    def forward(self, x):
        x = self.model_conv(x)
        x = self.model_linear(x)

        return x


def model_func(X_train, Y_train, X_test, Y_test, learning_rate=0.00001, num_epochs=10000, minibatch_size=64,
               print_cost=True,
               isPlot=True):
    # (m, n_H0, n_W0, n_C0) = X_train.shape  # (1080, 64, 64, 3)
    # n_y = Y_train.shape[1]  # 6
    train_cost_list = []
    train_acc_list = []
    test_cost_list = []
    test_acc_list = []

    # 转为数据集
    train_dataset = TensorDataset(X_train, Y_train)
    test_dataset = TensorDataset(X_test, Y_test)

    # minibatch
    train_loader = DataLoader(train_dataset, batch_size=minibatch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=True)

    # 初始化神经网络模型
    model = Net()
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)  # 优化器
    criterion = nn.CrossEntropyLoss()  # 计算cost的函数
    criterion.to(device)

    for epoch in range(num_epochs):
        train_cost = 0.0
        test_cost = 0.0
        train_acc = 0
        test_acc = 0
        train_batch_num = len(train_loader)
        test_batch_num = len(test_loader)

        # 开始训练
        model.train()
        for t, batch in enumerate(train_loader):
            minibatch_X, minibatch_Y = batch
            minibatch_X = minibatch_X.to(device)
            minibatch_Y = minibatch_Y.to(device)

            optimizer.zero_grad()  # 初始化梯度

            outputs = model(minibatch_X)  # 前向传播

            loss = criterion(outputs, minibatch_Y)  # 计算损失
            train_cost += loss.item() / train_batch_num

            loss.backward()  # 反向传播

            optimizer.step()  # 优化器更新

            _, predicted = torch.max(outputs.data, 1)
            train_acc += (predicted == minibatch_Y).sum().item()

        # 开始预测数据
        model.eval()
        with torch.no_grad():
            for t, batch in enumerate(test_loader):
                minibatch_X, minibatch_Y = batch
                minibatch_X = minibatch_X.to(device)
                minibatch_Y = minibatch_Y.to(device)

                optimizer.zero_grad()  # 初始化梯度

                outputs = model(minibatch_X)  # 前向传播

                loss = criterion(outputs, minibatch_Y)  # 计算损失
                test_cost += loss.item() / test_batch_num

                optimizer.step()  # 优化器更新

                _, predicted = torch.max(outputs.data, 1)
                test_acc += (predicted == minibatch_Y).sum().item()

        if print_cost and epoch % 100 == 0:
            print(f"--------------------第 {epoch + 1} 代------------------------------")
            print(f"训练集上,成本值为: {train_cost}\t正确率(%):{100 * train_acc / X_train.shape[0]}")
            print(f"测试及上,成本值为: {test_cost}\t正确率(%):{100 * test_acc / X_test.shape[0]}")

        train_cost_list.append(train_cost)
        train_acc_list.append(100 * train_acc / X_train.shape[0])
        test_cost_list.append(test_cost)
        test_acc_list.append(100 * test_acc / X_test.shape[0])

    if isPlot:
        plt.figure(1)
        plt.plot(np.squeeze(train_cost_list), 'blue')
        plt.plot(np.squeeze(test_cost_list), 'red')
        plt.legend(['Train Loss', 'Test Loss'], fontsize=14, loc='best')
        plt.xlabel('Epoch', fontsize=14)
        plt.ylabel('Loss', fontsize=14)
        plt.grid()
        plt.title("[LOSS] Learning rate =" + str(learning_rate))

        plt.figure(2)
        plt.plot(np.squeeze(train_acc_list), 'blue')
        plt.plot(np.squeeze(test_acc_list), 'red')
        plt.legend(['Train Accuracy', 'Test Accuracy'], fontsize=14, loc='best')
        plt.xlabel('Epoch', fontsize=14)
        plt.ylabel('Accuracy(%)', fontsize=14)
        plt.grid()
        plt.title("[accuracy] Learning rate =" + str(learning_rate))
        plt.show()

    return


model_func(X_train, Y_train, X_test, Y_test)
