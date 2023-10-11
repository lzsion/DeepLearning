# Author: LZS
# CreateTime: 2023/10/10  18:30
# FileName: C01W03_1
# Python Script

# 两层 多节点 神经网络

import numpy as np
import matplotlib.pyplot as plt
from testCases import *
import sklearn
import sklearn.datasets
import sklearn.linear_model
from sklearn.linear_model import LogisticRegressionCV
from planar_utils import plot_decision_boundary, sigmoid, load_planar_dataset, load_extra_datasets

X, Y = load_planar_dataset()  # 导入花图案
# X内存有花瓣图案的点坐标(x1, x2)
# Y代表点的颜色 (Y取值为0或1)

# print(X.shape)  # (2, 400)
# print(Y.shape)  # (1, 400)
# print(Y.reshape(X[0,:].shape).shape)  # (400,)

# 绘制花瓣图案，颜色由Y区分
# plt.scatter(X[0, :], X[1, :], c=Y.reshape(X[0, :].shape), s=40, cmap=plt.cm.Spectral)
# plt.show()

Y.reshape(X[0, :].shape)
clf = sklearn.linear_model.LogisticRegressionCV()
clf.fit(X.T, Y.ravel())


# 绘制决策边界
# plot_decision_boundary(lambda x: clf.predict(x), X, Y)
# plt.show()
# 结果不佳 引入神经网络

def layer_sizes(X, Y):
    """
    获得层的信息
    参数：
     X - 输入数据集,维度为（输入的数量，训练/测试的数量）
     Y - 标签，维度为（输出的数量，训练/测试数量）
    返回：
     n_x - 输入层的数量
     n_h - 隐藏层的节点数量
     n_y - 输出层的数量
    """

    # 输入层大小
    n_x = X.shape[0]
    # 隐藏层节点数
    n_h = 4
    # 输出层大小
    n_y = Y.shape[0]
    return (n_x, n_h, n_y)


def initialize_parameters(n_x, n_h, n_y):
    """
    初始化参数
    参数：
        n_x - 输入层节点的数量
        n_h - 隐藏层节点的数量
        n_y - 输出层节点的数量
    返回：
        parameters - 包含参数的字典：
            W1 - 权重矩阵,维度为（n_h，n_x）
            b1 - 偏向量，维度为（n_h，1）
            W2 - 权重矩阵，维度为（n_y，n_h）
            b2 - 偏向量，维度为（n_y，1）
    """

    # 设置一个种子，这样你的输出与我们的匹配，尽管初始化是随机的。
    np.random.seed(2)

    W1 = np.random.randn(n_h, n_x) * 0.01
    b1 = np.zeros((n_h, 1))
    W2 = np.random.randn(n_y, n_h) * 0.01
    b2 = np.zeros((n_y, 1))

    # 使用断言确保我的数据格式是正确的
    assert (W1.shape == (n_h, n_x))
    assert (b1.shape == (n_h, 1))
    assert (W2.shape == (n_y, n_h))
    assert (b2.shape == (n_y, 1))

    parameters = {"W1": W1,
                  "b1": b1,
                  "W2": W2,
                  "b2": b2}

    return parameters


def forward_propagation(X, parameters):
    """
    正向传播
    参数：
         X - 维度为（n_x，m）的输入数据。
         parameters - 初始化函数（initialize_parameters）的输出
    返回：
         A2 - 使用sigmoid()函数计算的第二次激活后的数值
         cache - 包含“Z1”，“A1”，“Z2”和“A2”的字典类型变量
     """

    # 从字典 “parameters” 中检索每个参数
    W1 = parameters["W1"]
    b1 = parameters["b1"]
    W2 = parameters["W2"]
    b2 = parameters["b2"]

    # 实现前向传播计算A2(概率)
    Z1 = np.dot(W1, X) + b1
    A1 = np.tanh(Z1)
    Z2 = np.dot(W2, A1) + b2
    A2 = sigmoid(Z2)

    # 使用断言确保我的数据格式是正确的
    assert (A2.shape == (1, X.shape[1]))

    cache = {"Z1": Z1,
             "A1": A1,
             "Z2": Z2,
             "A2": A2}

    return A2, cache


def compute_cost(A2, Y, parameters):
    """
    计算成本
    参数：
         A2 - 使用sigmoid()函数计算的第二次激活后的数值
         Y - "True"标签向量,维度为（1，数量）
         parameters - 一个包含W1，B1，W2和B2的字典类型的变量
    返回：
         成本 - 交叉熵成本给出方程（7）
    """

    # 样本数量
    m = Y.shape[1]

    # 计算交叉熵代价
    logprobs = Y * np.log(A2) + (1 - Y) * np.log(1 - A2)
    cost = -1 / m * np.sum(logprobs)

    # 确保损失是我们期望的维度
    # 例如，turns [[17]] into 17
    cost = np.squeeze(cost)

    assert (isinstance(cost, float))

    return cost


def backward_propagation(parameters, cache, X, Y):
    """
    反向传播
    参数：
     parameters - 包含我们的参数的一个字典类型的变量。
     cache - 包含“Z1”，“A1”，“Z2”和“A2”的字典类型的变量。
     X - 输入数据，维度为（2，数量）
     Y - “True”标签，维度为（1，数量）
    返回：
     grads - 包含W和b的导数一个字典类型的变量。
    """

    m = X.shape[1]

    # 首先，从字典“parameters”中检索W1和W2。
    W1 = parameters["W1"]
    W2 = parameters["W2"]

    # 还可以从字典“cache”中检索A1和A2。
    A1 = cache["A1"]
    A2 = cache["A2"]

    # 反向传播:计算 dW1、db1、dW2、db2。
    dZ2 = A2 - Y
    dW2 = 1 / m * np.dot(dZ2, A1.T)
    db2 = 1 / m * np.sum(dZ2, axis=1, keepdims=True)
    dZ1 = np.dot(W2.T, dZ2) * (1 - np.power(A1, 2))
    dW1 = 1 / m * np.dot(dZ1, X.T)
    db1 = 1 / m * np.sum(dZ1, axis=1, keepdims=True)

    grads = {"dW1": dW1,
             "db1": db1,
             "dW2": dW2,
             "db2": db2}

    return grads


def update_parameters(parameters, grads, learning_rate=1.2):
    """
    使用梯度下降更新参数
    参数：
     parameters - 包含参数的字典类型的变量。
     grads - 包含导数值的字典类型的变量。
     learning_rate - 学习速率
    返回：
     parameters - 包含更新参数的字典类型的变量。
    """

    # 从字典“parameters”中检索每个参数
    W1 = parameters["W1"]
    b1 = parameters["b1"]
    W2 = parameters["W2"]
    b2 = parameters["b2"]

    # 从字典“梯度”中检索每个梯度
    dW1 = grads["dW1"]
    db1 = grads["db1"]
    dW2 = grads["dW2"]
    db2 = grads["db2"]

    # 每个参数的更新规则
    W1 = W1 - learning_rate * dW1
    b1 = b1 - learning_rate * db1
    W2 = W2 - learning_rate * dW2
    b2 = b2 - learning_rate * db2

    parameters = {"W1": W1,
                  "b1": b1,
                  "W2": W2,
                  "b2": b2}

    return parameters


def nn_model(X, Y, n_h, num_iterations=10000, print_cost=False):
    """
    封装成一个函数
    参数：
        X - 数据集,维度为（2，示例数）
        Y - 标签，维度为（1，示例数）
        n_h - 隐藏层的数量
        num_iterations - 梯度下降循环中的迭代次数
        print_cost - 如果为True，则每1000次迭代打印一次成本数值
    返回：
        parameters - 模型学习的参数，它们可以用来进行预测。
     """

    # 初始化参数，然后检索 W1, b1, W2, b2。输入:“n_x, n_h, n_y”。
    n_x = layer_sizes(X, Y)[0]
    n_y = layer_sizes(X, Y)[2]

    # 初始化参数，然后检索 W1, b1, W2, b2。
    # 输入:“n_x, n_h, n_y”。输出=“W1, b1, W2, b2，参数”。
    parameters = initialize_parameters(n_x, n_h, n_y)
    W1 = parameters["W1"]
    b1 = parameters["b1"]
    W2 = parameters["W2"]
    b2 = parameters["b2"]

    # 循环(梯度下降)
    for i in range(0, num_iterations):

        # 前项传播
        A2, cache = forward_propagation(X, parameters)

        # 计算成本
        cost = compute_cost(A2, Y, parameters)

        # 反向传播
        grads = backward_propagation(parameters, cache, X, Y)

        # 更新参数
        parameters = update_parameters(parameters, grads)

        # 每1000次迭代打印成本
        if print_cost and i % 1000 == 0:
            print("Cost after iteration %i: %f" % (i, cost))

    return parameters


def predict(parameters, X):
    """
    使用学习的参数，为X中的每个示例预测一个类
    参数：
        parameters - 包含参数的字典类型的变量。
        X - 输入数据（n_x，m）
    返回
        predictions - 我们模型预测的向量（红色：0 /蓝色：1）
     """

    # 使用前向传播计算概率，并使用 0.5 作为阈值将其分类为 0/1。
    A2, cache = forward_propagation(X, parameters)
    predictions = np.round(A2)

    return predictions


'''
# 将X和Y放入模型训练
parameters = nn_model(X, Y, n_h=4, num_iterations=10000, print_cost=True)
# 绘制决策边界
plot_decision_boundary(lambda x: predict(parameters, x.T), X, Y)
plt.show()

# 正确率
predictions = predict(parameters, X)
accuracy = (np.dot(Y, predictions.T) + np.dot(1 - Y, 1 - predictions.T)) / float(Y.size) * 100
print('准确率: %d' % accuracy.item() + '%')
'''
# 调节隐藏层节点数量
plt.figure(figsize=(16, 32))
hidden_layer_sizes = [1, 2, 3, 4, 5, 10, 20]  # 隐藏层节点数量
for i, n_h in enumerate(hidden_layer_sizes):
    plt.subplot(5, 2, i + 1)
    plt.title('Hidden Layer of size %d' % n_h)
    parameters = nn_model(X, Y, n_h, num_iterations=5000)
    plot_decision_boundary(lambda x: predict(parameters, x.T), X, Y)
    predictions = predict(parameters, X)
    accuracy = float((np.dot(Y, predictions.T).item() + np.dot(1 - Y, 1 - predictions.T).item()) / float(Y.size) * 100)
    print("隐藏层的节点数量： {}  ，准确率: {} %".format(n_h, accuracy))
plt.show()
