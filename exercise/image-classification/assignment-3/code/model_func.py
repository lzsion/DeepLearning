# Author: LZS
# CreateTime: 2023/10/15  20:46
# FileName: model_func
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

from torch.utils.data import DataLoader, TensorDataset
from plot_func import plot_samples, plot_curve
from load_data import load_data
from CNN_Net import CNN
from ANN_Net import ANN


def model_CNN(train, test, learning_rate=0.01, epoch_num=100, device=torch.device('cpu'),
              print_cost=True, isPlot=True, isSaveFig=True, isSaveModel=True):
    model_name = 'CNN'

    model = CNN()  # 实例化网络
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()  # 损失函数
    criterion.to(device)

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)  # 定义优化器（训练参数，学习率）
    # optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)

    train_acc_list = []
    train_loss_list = []
    test_acc_list = []
    test_loss_list = []

    for epoch in range(epoch_num):
        train_loss = 0.0  # 训练集的loss
        test_loss = 0.0  # 测试集的loss

        train_acc = 0  # 训练集正确率
        test_acc = 0  # 测试集正确率

        train_sample_sum = 0  # 训练集样本总数
        test_sample_sum = 0  # 测试集样本总数

        # 开始训练
        model.train()
        for train_batch_idx, data in enumerate(train, 0):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)  # 转到gpu

            optimizer.zero_grad()  # 梯度清零 (初始化)

            outputs = model(inputs.float())  # 正向传播

            loss = criterion(outputs, labels.long())  # 计算loss
            train_loss += loss.item()  # 累加loss，后面求平均

            loss.backward()  # 反向传播

            optimizer.step()  # 优化器更新参数

            # 计算正确率
            _, predict = torch.max(outputs.data, 1)
            train_acc += (predict == labels).sum().item()  # 累加正确的个数
            train_sample_sum += predict.shape[0]  # 计算训练集样本总数，用于平均

        # 开始测试
        model.eval()
        with torch.no_grad():
            for test_batch_idx, data in enumerate(test, 0):
                inputs, labels = data
                inputs, labels = inputs.to(device), labels.to(device)  # 转到gpu

                outputs = model(inputs.float())  # 正向传播

                loss = criterion(outputs, labels.long())  # 计算loss
                test_loss += loss.item()  # 累加loss，后面求平均

                # 计算正确率
                _, predict = torch.max(outputs.data, 1)
                test_acc += (predict == labels).sum().item()  # 累加正确的个数
                test_sample_sum += predict.shape[0]  # 计算训练集样本总数，用于平均

        # if print_cost and epoch % 10 == 9:
        if print_cost:
            print(('-' * 20) + f"第 {epoch + 1} 代" + ('-' * 20))
            print(
                f"训练集上,成本值为: {train_loss / (train_batch_idx + 1)} \t 正确率(%):{100 * train_acc / train_sample_sum}")
            print(
                f"测试及上,成本值为: {test_loss / (test_batch_idx + 1)} \t 正确率(%):{100 * test_acc / test_sample_sum}")

        train_loss_list.append(train_loss / (train_batch_idx + 1))
        train_acc_list.append(100 * train_acc / train_sample_sum)
        test_acc_list.append(100 * test_acc / test_sample_sum)
        test_loss_list.append(test_loss / (test_batch_idx + 1))

    if isPlot:
        plot_curve(train_loss_list, test_loss_list, train_acc_list, test_acc_list, isSaveFig, model_name)
        plt.show()
    if isSaveModel:
        save_model(model, model_name)
        current_time = datetime.now()  # 获取当前系统时间
        formatted_time = current_time.strftime("%Y-%m-%d_%H-%M")  # 将时间格式化为字符串
        textpath = './model/MNIST_' + model_name + '_' + formatted_time + '.txt'
        images, labels = next(iter(train))
        minibatch_size = len(labels)
        with open(textpath, "w", encoding="utf-8") as file:
            sys.stdout = file  # 重定向标准输出到文件
            print(('-' * 20) + f"第 {epoch_num} 代" + ('-' * 20))
            print(
                f"训练集上,成本值为: {train_loss / (train_batch_idx + 1)} \t 正确率(%): {100 * train_acc / train_sample_sum}")
            print(
                f"测试及上,成本值为: {test_loss / (test_batch_idx + 1)} \t 正确率(%): {100 * test_acc / test_sample_sum}")
            print(f"学习率: {learning_rate}")
            print(f"minibatch_size={minibatch_size}")
            print(model)
            sys.stdout = sys.__stdout__  # 恢复标准输出
    return model


def model_ANN(train, test, learning_rate=0.01, epoch_num=100, device=torch.device('cpu'),
              print_cost=True, isPlot=True, isSaveFig=True, isSaveModel=True):
    model_name = 'ANN'

    model = ANN()  # 实例化网络
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()  # 损失函数
    criterion.to(device)

    # optimizer = optim.Adam(model.parameters(), lr=learning_rate)  # 定义优化器（训练参数，学习率）
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)

    train_acc_list = []
    train_loss_list = []
    test_acc_list = []
    test_loss_list = []

    for epoch in range(epoch_num):
        train_loss = 0.0  # 训练集的loss
        test_loss = 0.0  # 测试集的loss

        train_acc = 0  # 训练集正确率
        test_acc = 0  # 测试集正确率

        train_sample_sum = 0  # 训练集样本总数
        test_sample_sum = 0  # 测试集样本总数

        # 开始训练
        model.train()
        for train_batch_idx, data in enumerate(train, 0):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)  # 转到gpu

            optimizer.zero_grad()  # 梯度清零 (初始化)

            outputs = model(inputs.float())  # 正向传播

            loss = criterion(outputs, labels.long())  # 计算loss
            train_loss += loss.item()  # 累加loss，后面求平均

            loss.backward()  # 反向传播

            optimizer.step()  # 优化器更新参数

            # 计算正确率
            _, predict = torch.max(outputs.data, 1)
            train_acc += (predict == labels).sum().item()  # 累加正确的个数
            train_sample_sum += predict.shape[0]  # 计算训练集样本总数，用于平均

        # 开始测试
        model.eval()
        with torch.no_grad():
            for test_batch_idx, data in enumerate(test, 0):
                inputs, labels = data
                inputs, labels = inputs.to(device), labels.to(device)  # 转到gpu

                outputs = model(inputs.float())  # 正向传播

                loss = criterion(outputs, labels.long())  # 计算loss
                test_loss += loss.item()  # 累加loss，后面求平均

                # 计算正确率
                _, predict = torch.max(outputs.data, 1)
                test_acc += (predict == labels).sum().item()  # 累加正确的个数
                test_sample_sum += predict.shape[0]  # 计算训练集样本总数，用于平均

        # if print_cost and epoch % 10 == 9:
        if print_cost:
            print(('-' * 20) + f"第 {epoch + 1} 代" + ('-' * 20))
            print(
                f"训练集上,成本值为: {train_loss / (train_batch_idx + 1)} \t 正确率(%):{100 * train_acc / train_sample_sum}")
            print(
                f"测试及上,成本值为: {test_loss / (test_batch_idx + 1)} \t 正确率(%):{100 * test_acc / test_sample_sum}")

        train_loss_list.append(train_loss / (train_batch_idx + 1))
        train_acc_list.append(100 * train_acc / train_sample_sum)
        test_acc_list.append(100 * test_acc / test_sample_sum)
        test_loss_list.append(test_loss / (test_batch_idx + 1))

    if isPlot:
        plot_curve(train_loss_list, test_loss_list, train_acc_list, test_acc_list, isSaveFig, model_name)
        plt.show()
    if isSaveModel:
        save_model(model, model_name)
        current_time = datetime.now()  # 获取当前系统时间
        formatted_time = current_time.strftime("%Y-%m-%d_%H-%M")  # 将时间格式化为字符串
        textpath = './model/MNIST_' + model_name + '_' + formatted_time + '.txt'
        images, labels = next(iter(train))
        minibatch_size = len(labels)
        with open(textpath, "w", encoding="utf-8") as file:
            sys.stdout = file  # 重定向标准输出到文件
            print(('-' * 20) + f"第 {epoch_num} 代" + ('-' * 20))
            print(
                f"训练集上,成本值为: {train_loss / (train_batch_idx + 1)} \t 正确率(%): {100 * train_acc / train_sample_sum}")
            print(
                f"测试及上,成本值为: {test_loss / (test_batch_idx + 1)} \t 正确率(%): {100 * test_acc / test_sample_sum}")
            print(f"学习率: {learning_rate}")
            print(f"minibatch_size={minibatch_size}")
            print(model)
            sys.stdout = sys.__stdout__  # 恢复标准输出
    return model


def save_model(model, model_name):
    current_time = datetime.now()  # 获取当前系统时间
    formatted_time = current_time.strftime("%Y-%m-%d_%H-%M")  # 将时间格式化为字符串
    path = './model/MNIST_' + model_name + '_' + formatted_time + '.pth'
    torch.save(model, path)


def load_model(model_path, device=torch.device('cpu')):
    model = torch.load(model_path)
    model = model.to(device)
    return model
