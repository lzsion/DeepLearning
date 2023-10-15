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

fmap_block0 = {}
fmap_block1 = {}
fmap_block2 = {}


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
            print(model)
            sys.stdout = sys.__stdout__  # 恢复标准输出
    return model


def save_model(model):
    current_time = datetime.now()  # 获取当前系统时间
    formatted_time = current_time.strftime("%Y-%m-%d_%H-%M")  # 将时间格式化为字符串
    path = './model/ConvNet60_' + formatted_time + '.pth'
    torch.save(model, path)


def load_model(model_path, device=torch.device('cpu')):
    model = torch.load(model_path)
    model = model.to(device)
    return model


def model_visual(model, input_img, input_label, isSaveFig=False):
    print(model)

    model.for_ward[0].register_forward_hook(forward_hook0)
    model.for_ward[4].register_forward_hook(forward_hook1)
    model.for_ward[9].register_forward_hook(forward_hook2)
    outs = model(input_img)

    # print(fmap_block0['input'][0].shape, fmap_block0['output'][0].shape)
    # # torch.Size([1, 1, 60, 60]) torch.Size([16, 56, 56])
    # print(fmap_block1['input'][0].shape, fmap_block1['output'][0].shape)
    # # torch.Size([1, 16, 28, 28]) torch.Size([32, 22, 22])
    # print(fmap_block2['input'][0].shape, fmap_block2['output'][0].shape)
    # # torch.Size([1, 32, 11, 11]) torch.Size([64, 7, 7])

    fig = plt.figure(4)
    # plt.tight_layout()
    plt.imshow(input_img.cpu().reshape(60, 60), cmap='gray')
    plt.title("{}:{}".format('label', input_label.item()))
    plt.xticks([])
    plt.yticks([])
    if isSaveFig:
        plt.savefig('./fig/example_of_60_60')
    # plt.show()

    fig = plt.figure(5)
    for i in range(16):
        plt.subplot(4, 4, i + 1)
        plt.imshow(fmap_block0['output'][0][i].cpu().detach().numpy().reshape(56, 56), cmap='gray')
        plt.xticks([])  # 隐藏子图的 x 和 y 轴刻度
        plt.yticks([])
    plt.suptitle('Feature maps of [Conv1]16@5×5    label=' + str(input_label.item()))
    if isSaveFig:
        plt.savefig('./fig/60_60_conv1_feature')
    # plt.show()

    fig = plt.figure(6)
    for i in range(16):
        plt.subplot(4, 4, i + 1)
        plt.imshow(model.for_ward[0].weight[i].cpu().detach().numpy().reshape(5, 5), cmap='gray')
        plt.xticks([])
        plt.yticks([])
    plt.suptitle('Kernels of [Conv1]1×16@5×5')
    if isSaveFig:
        plt.savefig('./fig/60_60_conv1_kernals')
    # plt.show()

    fig = plt.figure(7)
    for i in range(32):
        plt.subplot(6, 6, i + 1)
        plt.imshow(fmap_block1['output'][0][i].cpu().detach().numpy().reshape(22, 22), cmap='gray')
        plt.xticks([])
        plt.yticks([])
    plt.suptitle('Feature maps of [Conv2]32@7×7    label=' + str(input_label.item()))
    if isSaveFig:
        plt.savefig('./fig/60_60_conv2_feature')
    # plt.show()

    fig = plt.figure(8)
    for i, j in enumerate(model.for_ward[4].weight[:, 0, :, :]):
        plt.subplot(6, 6, i + 1)
        plt.imshow(j.cpu().detach().numpy().reshape(7, 7), cmap='gray')
        plt.xticks([])
        plt.yticks([])
    plt.suptitle('(The first channel)Kernels of [Conv2]32×16@7×7')
    if isSaveFig:
        plt.savefig('./fig/60_60_conv2_kernals')
    # plt.show()

    fig = plt.figure(9)
    for i in range(64):
        plt.subplot(8, 8, i + 1)
        plt.imshow(fmap_block2['output'][0][i].cpu().detach().numpy().reshape(7, 7), cmap='gray')
        plt.xticks([])
        plt.yticks([])
    plt.suptitle('Feature maps of [Conv3]64@5×5    label=' + str(input_label.item()))
    if isSaveFig:
        plt.savefig('./fig/60_60_conv3_feature')
    # plt.show()

    fig = plt.figure(10)
    for i, j in enumerate(model.for_ward[9].weight[:, 0, :, :]):
        plt.subplot(8, 8, i + 1)
        plt.imshow(j.cpu().detach().numpy().reshape(5, 5), cmap='gray')
        plt.xticks([])
        plt.yticks([])
    plt.suptitle('(The first channel)Kernels of Conv3.64×32@5×5')
    if isSaveFig:
        plt.savefig('./fig/60_60_conv3_keranals')

    plt.show()


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
def forward_hook0(module, inp, outp):
    fmap_block0['input'] = inp
    fmap_block0['output'] = outp


def forward_hook1(module, inp, outp):
    fmap_block1['input'] = inp
    fmap_block1['output'] = outp


def forward_hook2(module, inp, outp):
    fmap_block2['input'] = inp
    fmap_block2['output'] = outp
