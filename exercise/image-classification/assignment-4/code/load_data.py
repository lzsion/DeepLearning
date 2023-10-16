# Author: LZS
# CreateTime: 2023/10/14  21:05
# FileName: load_data
# Python Script

# 加载数据


import torch
import scipy.io
from scipy.io import loadmat
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader


def load_data(data_path, minibatch_size=64):
    data = scipy.io.loadmat(data_path)  # 载入mat数据 为字典类型

    train_data = data['train_data']  # numpy.ndarray类型
    train_label = data['train_labels']
    test_data = data['test_data']
    test_label = data['test_labels']

    # print(train_data.shape, train_label.shape, test_data.shape, test_label.shape)
    # (3671, 60, 60) (1, 3671) (3203, 60, 60) (1, 3203)
    # (3671, 128, 128) (1, 3671) (3203, 128, 128) (1, 3203)

    train_data = train_data.reshape(train_data.shape[0], 1, train_data.shape[1], train_data.shape[2])  # 加上通道数
    test_data = test_data.reshape(test_data.shape[0], 1, test_data.shape[1], test_data.shape[2])

    train_data, train_label, test_data, test_label = map(torch.tensor, (
        train_data, train_label.squeeze(), test_data, test_label.squeeze()))
    # 使用map函数将torch.tensor应用于四个数据集，即都转为torch.Tensor张量

    # print(train_data.shape, train_label.shape, test_data.shape, test_label.shape)
    # torch.Size([3671, 1, 60, 60]) torch.Size([3671]) torch.Size([3203, 1, 60, 60]) torch.Size([3203])

    # 数据集标准化
    mean = train_data.mean()  # 求训练集的均值
    std = train_data.std()  # 求训练集的方差

    # print(mean, std)
    # tensor(0.0727, dtype=torch.float64) tensor(0.1475, dtype=torch.float64)

    train_data = (train_data - mean) / std  # 标准化
    test_data = (test_data - mean) / std  # 测试集也使用训练集均值和方差标准化

    # print(train_data.mean(), train_data.std(), test_data.mean(), test_data.std())
    # tensor(2.0921e-17, dtype=torch.float64) tensor(1.0000, dtype=torch.float64)
    # tensor(-0.0328, dtype=torch.float64) tensor(0.9620, dtype=torch.float64)

    # minibatch_size = 64  # 设置minibatch大小
    # 转为Tensor数据集
    train_xy = TensorDataset(train_data, train_label)
    test_xy = TensorDataset(test_data, test_label)

    train = DataLoader(train_xy, batch_size=minibatch_size, shuffle=True)
    test = DataLoader(test_xy, batch_size=len(test_xy))

    # print(len(train_xy), len(train), len(test_xy), len(test))
    # 3671 58 3203 13

    return train, test


def get_first_data(train, device=torch.device('cpu')):
    batch_img, batch_label = next(iter(train))  # 获取第一个batch
    input_img = batch_img[0].view(1, 1, 60, 60).to(device).float()
    input_label = batch_label[0].view(1).to(device)
    return input_img, input_label
