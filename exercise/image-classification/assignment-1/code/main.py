# Author: LZS
# CreateTime: 2023/10/14  15:48
# FileName: main
# Python Script

# pytorch构建LeNet模型

import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import time

from PIL import Image
from torch.utils.data import DataLoader, TensorDataset


class LeNet(nn.Module):  # 继承于nn.Module这个父类
    def __init__(self):  # 初始化网络结构
        super(LeNet, self).__init__()  # 多继承需用到super函数
        self.model_conv = nn.Sequential(
            nn.Conv2d(3, 16, 5),  # (3, 32, 32) -> (16, 28, 28)
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # -> (16, 14, 14)

            nn.Conv2d(16, 32, 5),  # -> (32, 10, 10)
            nn.ReLU(),
            nn.MaxPool2d(2, 2)  # -> (32, 5, 5)
        )
        self.model_linear = nn.Sequential(
            nn.Linear(32 * 5 * 5, 120),  # -> (120)
            nn.ReLU(),
            nn.Linear(120, 84),  # -> (84)
            nn.ReLU(),
            nn.Linear(84, 10)  # -> (10)
        )

    def forward(self, x):  # 正向传播过程
        x = self.model_conv(x)
        x = x.view(-1, 32 * 5 * 5)  # -> (32*5*5)
        x = self.model_linear(x)
        return x


transform = transforms.Compose([transforms.ToTensor(),  # 转成Tensor类型
                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])  # 归一化
classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
# 导入50000张训练图片
train_set = torchvision.datasets.CIFAR10(root='./data',  # 数据集存放目录
                                         train=True,  # 表示是数据集中的训练集
                                         download=False,  # 第一次运行时为True，下载数据集，下载完成后改为False
                                         transform=transform)  # 预处理过程
# 加载训练集，实际过程需要分批次（batch）训练
train_loader = torch.utils.data.DataLoader(train_set,  # 导入的训练集
                                           batch_size=50,  # 每批训练的样本数
                                           shuffle=True,  # 是否打乱训练集
                                           num_workers=0)  # 使用线程数，在windows下设置为0
# 导入10000张测试图片
test_set = torchvision.datasets.CIFAR10(root='./data',
                                        train=False,  # 表示是数据集中的测试集
                                        download=False,  # 第一次运行时为True，下载数据集，下载完成后改为False
                                        transform=transform)
# 加载测试集
test_loader = torch.utils.data.DataLoader(test_set,
                                          batch_size=10000,  # 每批用于验证的样本数
                                          shuffle=False, num_workers=0)  # 使用线程数，在windows下设置为0


# 获取测试集中的图像和标签，用于后续accuracy计算
# train_data_iter = iter(train_loader)
# test_data_iter = iter(test_loader)
# test_image, test_label = test_data_iter.next()
# for t,(test_image, test_label) in enumerate(test_loader):


# print(test_image.shape)
# print(test_label.shape)

# for images, labels in train_loader:
#     print(images.shape)  # torch.Size([50, 3, 32, 32])
#     print(labels.shape)  # torch.Size([50])

def imshow(img, label, num=4):  # 预览数据集的图片以及label
    img = torchvision.utils.make_grid(img[:num])
    img = img / 2 + 0.5
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.axis("off")
    plt.title((' ' * 15).join('%5s' % classes[label[j]] for j in range(num)))
    plt.show()


# for images, labels in train_loader:
#     imshow(images, labels)

#     print(' '.join('%5s' % classes[labels[j]] for j in range(4)))


def model_train(train_loader, test_loader, device=torch.device("cpu"), learning_rate=0.001, epoch_num=5,
                save_path='./LeNet.pth'):
    (test_image, test_label) = next(iter(test_loader))
    test_image = test_image.to(device)
    test_label = test_label.to(device)

    net = LeNet()  # 定义训练的网络模型
    net.to(device)

    loss_function = nn.CrossEntropyLoss()  # 定义损失函数为交叉熵损失函数
    loss_function.to(device)

    optimizer = optim.Adam(net.parameters(), lr=learning_rate)  # 定义优化器（训练参数，学习率）

    for epoch in range(epoch_num):  # 一个epoch即对整个训练集进行一次训练
        running_loss = 0.0  # 初始化loss
        time_start = time.perf_counter()  # 计算时间

        for step, data in enumerate(train_loader, start=0):  # 遍历训练集，step从0开始计算
            inputs, labels = data  # 获取训练集的图像和标签
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()  # 清除历史梯度 梯度初始化

            # forward + backward + optimize
            outputs = net(inputs)  # 正向传播

            loss = loss_function(outputs, labels)  # 计算损失

            loss.backward()  # 反向传播

            optimizer.step()  # 优化器更新参数

            # 打印耗时、损失、准确率
            running_loss += loss.item()
            if step % 500 == 499:  # print every 1000 mini-batches，每1000步打印一次
                with torch.no_grad():  # 在以下步骤中（验证过程中）不用计算每个节点的损失梯度，防止内存占用
                    outputs = net(test_image)  # 测试集传入网络（test_batch_size=10000），output维度为[10000,10]
                    predict_y = torch.max(outputs, dim=1)[1]  # 以output中值最大位置对应的索引（标签）作为预测输出
                    accuracy = (predict_y == test_label).sum().item() / test_label.size(0)

                    print(f'epoch: {epoch + 1} ,\t step: {step + 1:5d}')
                    print(f'train_loss: {running_loss / 500:.3f} ,\t test_accuracy: {accuracy:.3f}')
                    print(f'time = {time.perf_counter() - time_start:.3f} s')
                    running_loss = 0.0

    print('Finished Training')

    # 保存训练得到的参数
    # save_path = './LeNet.pth'
    torch.save(net, save_path)


def model_predict(img_path, save_path='./LeNet.pth', device=torch.device("cpu")):
    model = torch.load(save_path)  # 加载模型
    model.to(device)
    # img_path = '1.jpg'
    img_data = np.array(Image.open(img_path).resize((32, 32)))  # 变换大小
    img_data = transform(img_data)  # 转换(归一化)
    img_data = torch.unsqueeze(img_data, dim=0)  # 增加一个维度
    img_data = img_data.to(device)
    out = model(img_data)
    label = out.data.max(1, keepdims=True)[1]
    classes = ('plane', 'car', 'bird', 'cat',
               'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    print("the category is : " + classes[label])


device = torch.device("cuda")
save_path = './LeNet.pth'

# model_train(train_loader, test_loader, device=device, epoch_num=20, save_path=save_path)

model_predict('./imgs/1.jpg', save_path=save_path, device=device)
