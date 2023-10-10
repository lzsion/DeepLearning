# Author: LZS
# CreateTime: 2023/10/10  17:43
# FileName: C01W01_1
# Python Script

# 单层 单节点 神经网络

import numpy as np
import matplotlib.pyplot as plt
import h5py
import scipy
from PIL import Image
from scipy import ndimage

from load_dataset import load_dataset

# 调用load_dataset.py中的load_dataset()函数加载数据
train_set_x_orig, train_set_y, test_set_x_orig, test_set_y, classes = load_dataset()

# plt.imshow(train_set_x_orig[100])
# plt.show()

# m_train = train_set_x_orig.shape[0]   # 训练集里图片的数量。
# m_test = test_set_x_orig.shape[0]     # 测试集里图片的数量。
# num_px = train_set_x_orig.shape[1]    # 训练集里图片的宽度
# num_py = train_set_x_orig.shape[2]    # 训练集里图片的宽度

# 重塑维度
train_set_x_flatten = train_set_x_orig.reshape(train_set_x_orig.shape[0], -1).T
# 将测试集的维度降低并转置。
test_set_x_flatten = test_set_x_orig.reshape(test_set_x_orig.shape[0], -1).T

# print(train_set_x_flatten.shape)
# print(test_set_x_flatten.shape)

# 标准化
train_set_x = train_set_x_flatten / 255
test_set_x = test_set_x_flatten / 255


def sigmoid(x):  # 定义sigmoid激活函数
    s = 1 / (1 + np.exp(-x))
    return s


def initialize_with_zeros(dim):  # 初始化参数
    """
    此函数为w创建一个维度为（dim，1）的0向量，并将b初始化为0。
    参数：
        dim  - 我们想要的w矢量的大小（或者这种情况下的参数数量）

    返回：
        w  - 维度为（dim，1）的初始化向量。
        b  - 初始化的标量（对应于偏差）
    """
    w = np.zeros((dim, 1))
    b = 0

    # 使用断言来确保我要的数据是正确的
    # w 的维度是 (dim,1)
    assert (w.shape == (dim, 1))
    # b 的类型是 float 或者是 int
    assert (isinstance(b, float) or isinstance(b, int))

    return w, b


# dim = 2
# w, b = initialize_with_zeros(dim)
# print("w = " + str(w))
# print("b = " + str(b))

def propagate(w, b, X, Y):  # 前向传播和反向传播
    """
    实现前向和后向传播的传播函数，计算成本函数及其梯度。
    参数：
        w  - 权重，大小不等的数组（num_px * num_px * 3，1）
        b  - 偏差，一个标量
        X  - 矩阵类型为（num_px * num_px * 3，训练数量）
        Y  - 真正的“标签”矢量（如果非猫则为0，如果是猫则为1），矩阵维度为(1,训练数据数量)
    返回：
        cost- 逻辑回归的负对数似然成本
        dw  - 相对于w的损失梯度，因此与w相同的形状
        db  - 相对于b的损失梯度，因此与b的形状相同
    """

    m = X.shape[1]

    # 正向传播
    # 计算激活函数
    # 这里是单层单节点
    A = sigmoid(np.dot(w.T, X) + b)
    # 计算成本
    cost = -1 / m * np.sum(Y * np.log(A) + (1 - Y) * np.log(1 - A))  # compute cost

    # 反向传播
    dw = 1 / m * np.dot(X, (A - Y).T)
    db = 1 / m * np.sum(A - Y)

    # 使用断言确保我的数据是正确的
    assert (dw.shape == w.shape)
    assert (db.dtype == float)
    cost = np.squeeze(cost)
    assert (cost.shape == ())

    # 创建一个字典，把 dw 和 db 保存起来。
    grads = {"dw": dw,
             "db": db}

    return grads, cost


def optimize(w, b, X, Y, num_iterations, learning_rate, print_cost=False):  # 梯度下降优化参数
    """
    此函数通过运行梯度下降算法来优化w和b
    参数：
        w  - 权重，大小不等的数组（num_px * num_px * 3，1）
        b  - 偏差，一个标量
        X  - 维度为（num_px * num_px * 3，训练数据的数量）的数组。
        Y  - 真正的“标签”矢量（如果非猫则为0，如果是猫则为1），矩阵维度为(1,训练数据的数量)
        num_iterations  - 优化循环的迭代次数
        learning_rate  - 梯度下降更新规则的学习率
        print_cost  - 每100步打印一次损失值
    返回：
        params  - 包含权重w和偏差b的字典
        grads  - 包含权重和偏差相对于成本函数的梯度的字典
        成本 - 优化期间计算的所有成本列表，将用于绘制学习曲线。
    提示：
    我们需要写下两个步骤并遍历它们：
        1）计算当前参数的成本和梯度，使用propagate（）。
        2）使用w和b的梯度下降法则更新参数。
    """

    costs = []

    for i in range(num_iterations):

        grads, cost = propagate(w, b, X, Y)

        dw = grads["dw"]
        db = grads["db"]

        w = w - learning_rate * dw
        b = b - learning_rate * db

        # 记录成本
        if i % 100 == 0:
            costs.append(cost)

        # 每 100 次训练 打印成本
        if print_cost and i % 100 == 0:
            print("Cost after iteration %i: %f" % (i, cost))

    params = {"w": w,
              "b": b}

    grads = {"dw": dw,
             "db": db}

    return params, grads, costs


def predict(w, b, X):  # 利用w和b参数计算y
    """
    使用学习逻辑回归参数 logistic(w，b) 预测标签是0还是1，
    参数：
        w  - 权重，大小不等的数组（num_px * num_px * 3，1）
        b  - 偏差，一个标量
        X  - 维度为（num_px * num_px * 3，训练数据的数量）的数据
    返回：
        Y_prediction  - 包含X中所有图片的所有预测【0 | 1】的一个numpy数组（向量）

    """

    # 图片的数量
    m = X.shape[1]
    Y_prediction = np.zeros((1, m))
    w = w.reshape(X.shape[0], 1)

    # 预测猫在图片中出现的概率
    A = sigmoid(np.dot(w.T, X) + b)

    for i in range(A.shape[1]):
        # 将概率 a[0，i] 转换为实际预测 p[0，i]
        if A[0, i] <= 0.5:
            Y_prediction[0, i] = 0
        else:
            Y_prediction[0, i] = 1

    # 使用断言
    assert (Y_prediction.shape == (1, m))

    return Y_prediction


def model(X_train, Y_train, X_test, Y_test, num_iterations=2000, learning_rate=0.5, print_cost=False):
    """
    通过调用之前实现的函数来构建逻辑回归模型
    参数：
        X_train  - numpy的数组,维度为（num_px * num_px * 3，m_train）的训练集
        Y_train  - numpy的数组,维度为（1，m_train）（矢量）的训练标签集
        X_test   - numpy的数组,维度为（num_px * num_px * 3，m_test）的测试集
        Y_test   - numpy的数组,维度为（1，m_test）的（向量）的测试标签集
        num_iterations  - 表示用于优化参数的迭代次数的超参数
        learning_rate  - 表示optimize（）更新规则中使用的学习速率的超参数
        print_cost  - 设置为true以每100次迭代打印成本

    返回：
        d  - 包含有关模型信息的字典。
    """

    # 初始化全零参数
    w, b = initialize_with_zeros(X_train.shape[0])

    # 梯度下降
    parameters, grads, costs = optimize(w, b, X_train, Y_train, num_iterations, learning_rate, print_cost)

    # 从“parameters”字典中检索参数w和b
    w = parameters["w"]
    b = parameters["b"]

    # 预测测试/训练集的例子
    Y_prediction_test = predict(w, b, X_test)
    Y_prediction_train = predict(w, b, X_train)

    # 打印训练后的准确性
    print("train accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction_train - Y_train)) * 100))
    print("test accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction_test - Y_test)) * 100))

    d = {"costs": costs,
         "Y_prediction_test": Y_prediction_test,
         "Y_prediction_train": Y_prediction_train,
         "w": w,
         "b": b,
         "learning_rate": learning_rate,
         "num_iterations": num_iterations}

    return d


# d = model(train_set_x, train_set_y, test_set_x, test_set_y, num_iterations=10000, learning_rate=0.001, print_cost=True)

# # 绘制损失曲线
# costs = np.squeeze(d['costs'])  # 从字典 d 中提取出一个名为 'costs' 的项
# plt.plot(costs)
# plt.ylabel('cost')
# plt.xlabel('iterations (per hundreds)')
# plt.title("Learning rate =" + str(d["learning_rate"]))
# plt.show()

learning_rates = [0.01, 0.001, 0.0001]
models = {}
for i in learning_rates:
    print("学习率: " + str(i))
    models[str(i)] = model(train_set_x, train_set_y, test_set_x, test_set_y, num_iterations = 5000, learning_rate = i, print_cost = False)
    print('\n' + "-------------------------------------------------------" + '\n')

for i in learning_rates:
    plt.plot(np.squeeze(models[str(i)]["costs"]), label= str(models[str(i)]["learning_rate"]))

plt.ylabel('cost')
plt.xlabel('iterations')

legend = plt.legend(loc='upper center', shadow=True)
frame = legend.get_frame()
frame.set_facecolor('0.90')
plt.show()
