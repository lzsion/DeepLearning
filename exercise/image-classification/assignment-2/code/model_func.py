# Author: LZS
# CreateTime: 2023/10/14  22:00
# FileName: model_func
# Python Script
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch import optim
from MyNet import ConvNet
from load_data import load_data
from datetime import datetime
from plot_func import plot_curve


def model_func(train, test, learning_rate=0.02, num_epochs=100, device=torch.device('cpu'),
               print_cost=True, isPlot=True, isSaveFig=True, isSaveModel=True):
    model = ConvNet()  # 实例化网络
    model = model.to(device)

    criterion = nn.CrossEntropyLoss(reduction='mean').to(device)  # cost函数

    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)  # 优化器 (momentum)

    # print(model)

    # 初始化训练集测试集上正确率和loss，用于绘图
    train_acc_list = []
    train_loss_list = []
    test_acc_list = []
    test_loss_list = []

    for epoch in range(num_epochs):
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

        if print_cost and epoch % 10 == 9:
            print(('-' * 20) + f"第 {epoch + 1} 代" + ('-' * 20))
            print(
                f"训练集上,成本值为: {train_loss / train_batch_idx} \t 正确率(%):{100 * train_acc / train_sample_sum}")
            print(f"测试及上,成本值为: {test_loss / test_batch_idx} \t 正确率(%):{100 * test_acc / test_sample_sum}")

        train_loss_list.append(train_loss / train_batch_idx)
        train_acc_list.append(100 * train_acc / train_sample_sum)
        test_acc_list.append(100 * test_acc / test_sample_sum)
        test_loss_list.append(test_loss / test_batch_idx)

    if isPlot:
        plot_curve(train_loss_list, test_loss_list, train_acc_list, test_acc_list, isSaveFig)
        plt.show()
    if isSaveModel:
        save_model(model)
        current_time = datetime.now()  # 获取当前系统时间
        formatted_time = current_time.strftime("%Y-%m-%d_%H-%M")  # 将时间格式化为字符串
        textpath = './model/ConvNet60_' + formatted_time + '.txt'
        with open(textpath, "w", encoding="utf-8") as file:
            sys.stdout = file  # 重定向标准输出到文件
            print(('-' * 20) + f"第 {num_epochs} 代" + ('-' * 20))
            print(
                f"训练集上,成本值为: {train_loss / train_batch_idx} \t 正确率(%): {100 * train_acc / train_sample_sum}")
            print(f"测试及上,成本值为: {test_loss / test_batch_idx} \t 正确率(%): {100 * test_acc / test_sample_sum}")
            print(f"学习率: {learning_rate}")
            sys.stdout = sys.__stdout__  # 恢复标准输出


def save_model(model):
    current_time = datetime.now()  # 获取当前系统时间
    formatted_time = current_time.strftime("%Y-%m-%d_%H-%M")  # 将时间格式化为字符串
    path = './model/ConvNet60_' + formatted_time + '.pth'
    torch.save(model, path)



# def plot_curve(train_loss_list, test_loss_list, train_acc_list, test_acc_list, isSaveFig):
#     current_time = datetime.now()  # 获取当前系统时间
#     formatted_time = current_time.strftime("%Y-%m-%d_%H-%M")  # 将时间格式化为字符串
#     loss_path = './fig/fig-LOSS_' + formatted_time
#     acc_path = './fig/fig-Accuracy_' + formatted_time
#
#     fig = plt.figure(2)
#     plt.plot(range(len(train_loss_list)), train_loss_list, 'blue')
#     plt.plot(range(len(test_loss_list)), test_loss_list, 'red')
#     plt.legend(['Train Loss', 'Test Loss'], fontsize=14, loc='best')
#     plt.xlabel('Epoch', fontsize=14)
#     plt.ylabel('Loss', fontsize=14)
#     plt.grid()
#     if isSaveFig:
#         plt.savefig(loss_path)
#     # plt.show()
#
#     fig = plt.figure(3)
#     plt.plot(range(len(train_acc_list)), train_acc_list, 'blue')
#     plt.plot(range(len(test_acc_list)), test_acc_list, 'red')
#     plt.legend(['Train Accuracy', 'Test Accuracy'], fontsize=14, loc='best')
#     plt.xlabel('Epoch', fontsize=14)
#     plt.ylabel('Accuracy(%)', fontsize=14)
#     plt.grid()
#     if isSaveFig:
#         plt.savefig(acc_path)
#     # plt.show()
