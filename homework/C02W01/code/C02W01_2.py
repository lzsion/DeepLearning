# Author: LZS
# CreateTime: 2023/10/11  14:27
# FileName: C02W01_1
# Python Script

# 是否使用正则化的影响

import numpy as np
import matplotlib.pyplot as plt
from reg_utils import sigmoid, relu, plot_decision_boundary, initialize_parameters, load_2D_dataset, predict_dec
from reg_utils import compute_cost, predict, forward_propagation, backward_propagation, update_parameters
import sklearn
import sklearn.datasets
import scipy.io
from testCases import *

plt.rcParams['figure.figsize'] = (7.0, 4.0)  # set default size of plots
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

train_X, train_Y, test_X, test_Y = load_2D_dataset()


# plt.show()


def compute_cost_with_regularization(A3, Y, parameters, lambd):
    """
    实现公式2的L2正则化计算成本
    参数：
        A3 - 正向传播的输出结果，维度为（输出节点数量，训练/测试的数量）
        Y - 标签向量，与数据一一对应，维度为(输出节点数量，训练/测试的数量)
        parameters - 包含模型学习后的参数的字典
    返回：
        cost - 使用公式2计算出来的正则化损失的值
    """

    m = Y.shape[1]
    W1 = parameters["W1"]
    W2 = parameters["W2"]
    W3 = parameters["W3"]

    cross_entropy_cost = compute_cost(A3, Y)

    L2_regularization_cost = lambd * (np.sum(np.square(W1)) + np.sum(np.square(W2)) + np.sum(np.square(W3))) / (2 * m)

    cost = cross_entropy_cost + L2_regularization_cost

    return cost


# 当然，因为改变了成本函数，我们也必须改变向后传播的函数，
# 所有的梯度都必须根据这个新的成本值来计算。
def backward_propagation_with_regularization(X, Y, cache, lambd):
    """
    实现我们添加了L2正则化的模型的后向传播。
    参数：
        X - 输入数据集，维度为（输入节点数量，数据集里面的数量）
        Y - 标签，维度为（输出节点数量，数据集里面的数量）
        cache - 来自forward_propagation（）的cache输出
        lambda - regularization超参数，实数
    返回：
        gradients - 一个包含了每个参数、激活值和预激活值变量的梯度的字典
    """

    m = X.shape[1]

    (Z1, A1, W1, b1, Z2, A2, W2, b2, Z3, A3, W3, b3) = cache

    dZ3 = A3 - Y

    dW3 = (1 / m) * np.dot(dZ3, A2.T) + ((lambd * W3) / m)
    db3 = (1 / m) * np.sum(dZ3, axis=1, keepdims=True)

    dA2 = np.dot(W3.T, dZ3)
    dZ2 = np.multiply(dA2, np.int64(A2 > 0))
    dW2 = (1 / m) * np.dot(dZ2, A1.T) + ((lambd * W2) / m)
    db2 = (1 / m) * np.sum(dZ2, axis=1, keepdims=True)

    dA1 = np.dot(W2.T, dZ2)
    dZ1 = np.multiply(dA1, np.int64(A1 > 0))
    dW1 = (1 / m) * np.dot(dZ1, X.T) + ((lambd * W1) / m)
    db1 = (1 / m) * np.sum(dZ1, axis=1, keepdims=True)

    gradients = {"dZ3": dZ3, "dW3": dW3, "db3": db3, "dA2": dA2,
                 "dZ2": dZ2, "dW2": dW2, "db2": db2, "dA1": dA1,
                 "dZ1": dZ1, "dW1": dW1, "db1": db1}

    return gradients


def forward_propagation_with_dropout(X, parameters, keep_prob=0.5):
    """
    带有dropout的前向传播 : LINEAR -> RELU + DROPOUT -> LINEAR -> RELU + DROPOUT -> LINEAR -> SIGMOID.
    参数:
        X -- 输入数据集 (2, m)
        parameters -- 包含"W1", "b1", "W2", "b2", "W3", "b3"的字典:
            W1 -- weight matrix of shape (20, 2)
            b1 -- bias vector of shape (20, 1)
            W2 -- weight matrix of shape (3, 20)
            b2 -- bias vector of shape (3, 1)
            W3 -- weight matrix of shape (1, 3)
            b3 -- bias vector of shape (1, 1)
        keep_prob - 保留节点的概率
    返回:
    A3 -- 输出层的值 (1,1)
    cache -- tuple, 缓存值
    """

    np.random.seed(1)

    # retrieve parameters
    W1 = parameters["W1"]
    b1 = parameters["b1"]
    W2 = parameters["W2"]
    b2 = parameters["b2"]
    W3 = parameters["W3"]
    b3 = parameters["b3"]

    # LINEAR -> RELU -> LINEAR -> RELU -> LINEAR -> SIGMOID
    Z1 = np.dot(W1, X) + b1
    A1 = relu(Z1)
    # 第一层不用dropout
    # START CODE HERE ### (approx. 4 lines)         # Steps 1-4 below correspond to the Steps 1-4 described above.
    D1 = np.random.rand(A1.shape[0], A1.shape[1])  # Step 1: initialize matrix D1 = np.random.rand(..., ...)
    D1 = D1 < keep_prob  # Step 2: convert entries of D1 to 0 or 1 (using keep_prob as the threshold)
    A1 = A1 * D1  # Step 3: shut down some neurons of A1
    A1 = A1 / keep_prob  # Step 4: scale the value of neurons that haven't been shut down
    # END CODE HERE ###
    Z2 = np.dot(W2, A1) + b2
    A2 = relu(Z2)
    # START CODE HERE ### (approx. 4 lines)
    D2 = np.random.rand(A2.shape[0], A2.shape[1])  # Step 1: initialize matrix D2 = np.random.rand(..., ...)
    D2 = D2 < keep_prob  # Step 2: convert entries of D2 to 0 or 1 (using keep_prob as the threshold)
    A2 = A2 * D2  # Step 3: shut down some neurons of A2
    A2 = A2 / keep_prob  # Step 4: scale the value of neurons that haven't been shut down
    # END CODE HERE ###
    Z3 = np.dot(W3, A2) + b3
    A3 = sigmoid(Z3)

    cache = (Z1, D1, A1, W1, b1, Z2, D2, A2, W2, b2, Z3, A3, W3, b3)

    return A3, cache


def backward_propagation_with_dropout(X, Y, cache, keep_prob):
    """
    带有dropout的反向传播
    参数:
        X -- 输入数据集(2, m)
        Y -- 标签(output size, number of examples)
        cache -- 从forward_propagation_with_dropout()得到的缓存
        keep_prob - 保留节点的概率
    返回:
        gradients -- 一个包含了每个参数、激活值和预激活值变量的梯度的字典
    """

    m = X.shape[1]
    (Z1, D1, A1, W1, b1, Z2, D2, A2, W2, b2, Z3, A3, W3, b3) = cache

    dZ3 = A3 - Y
    dW3 = 1. / m * np.dot(dZ3, A2.T)
    db3 = 1. / m * np.sum(dZ3, axis=1, keepdims=True)
    dA2 = np.dot(W3.T, dZ3)
    # 使用dropout，即使部分节点的A值置零
    # 需要将dropout的节点导数置零，使用cache中的D1，D2
    # START CODE HERE ### (≈ 2 lines of code)
    dA2 = dA2 * D2  # Step 1: 用 D2 将dropout的节点置零
    dA2 = dA2 / keep_prob  # Step 2: 将未置零的节点归一化
    # END CODE HERE ###
    dZ2 = np.multiply(dA2, np.int64(A2 > 0))
    dW2 = 1. / m * np.dot(dZ2, A1.T)
    db2 = 1. / m * np.sum(dZ2, axis=1, keepdims=True)

    dA1 = np.dot(W2.T, dZ2)
    # START CODE HERE ### (≈ 2 lines of code)
    dA1 = dA1 * D1  # Step 1: 用 D1 将dropout的节点置零
    dA1 = dA1 / keep_prob  # Step 2: 将未置零的节点归一化
    # END CODE HERE ###
    dZ1 = np.multiply(dA1, np.int64(A1 > 0))
    dW1 = 1. / m * np.dot(dZ1, X.T)
    db1 = 1. / m * np.sum(dZ1, axis=1, keepdims=True)

    gradients = {"dZ3": dZ3, "dW3": dW3, "db3": db3, "dA2": dA2,
                 "dZ2": dZ2, "dW2": dW2, "db2": db2, "dA1": dA1,
                 "dZ1": dZ1, "dW1": dW1, "db1": db1}

    return gradients


def model(X, Y, learning_rate=0.05, num_iterations=30000, print_cost=True, is_plot=True, lambd=0, keep_prob=1):
    """
    实现一个三层的神经网络：LINEAR ->RELU -> LINEAR -> RELU -> LINEAR -> SIGMOID
    参数：
        X - 输入的数据，维度为(2, 要训练/测试的数量)
        Y - 标签，【0(蓝色) | 1(红色)】，维度为(1，对应的是输入的数据的标签)
        learning_rate - 学习速率
        num_iterations - 迭代的次数
        print_cost - 是否打印成本值，每迭代10000次打印一次，但是每1000次记录一个成本值
        is_polt - 是否绘制梯度下降的曲线图
        lambd - 正则化的超参数，实数
        keep_prob - 随机删除节点的概率
    返回
        parameters - 学习后的参数
    """

    grads = {}
    costs = []
    m = X.shape[1]
    layers_dims = [X.shape[0], 20, 3, 1]

    # 初始化参数(使用he方法)
    parameters = initialize_parameters(layers_dims)

    # 开始学习
    for i in range(0, num_iterations):
        # 前向传播
        # 是否随机删除节点
        if keep_prob == 1:
            # 不随机删除节点
            a3, cache = forward_propagation(X, parameters)
        elif keep_prob < 1:
            # 随机删除节点
            a3, cache = forward_propagation_with_dropout(X, parameters, keep_prob)
        else:
            print("keep_prob参数错误！程序退出。")
            exit

        # 计算成本
        # 是否正则化
        if lambd == 0:
            # 不使用 L2 正则化
            cost = compute_cost(a3, Y)
        else:
            # 使用 L2 正则化
            cost = compute_cost_with_regularization(a3, Y, parameters, lambd)

        # 反向传播
        # 可以同时使用 L2 正则化和随机删除节点，但是本次实验不同时使用。
        assert (lambd == 0 or keep_prob == 1)

        # 两个参数的使用情况
        if lambd == 0 and keep_prob == 1:
            # 不使用 L2 正则化和不使用随机删除节点
            grads = backward_propagation(X, Y, cache)
        elif lambd != 0:
            # 使用 L2 正则化，不使用随机删除节点
            grads = backward_propagation_with_regularization(X, Y, cache, lambd)
        elif keep_prob < 1:
            # 使用随机删除节点，不使用 L2 正则化
            grads = backward_propagation_with_dropout(X, Y, cache, keep_prob)

        # 仅使用正则化，正向传播不变，计算cost方式改变，反向传播函数改变
        # 仅使用dropout，正向传播需要改变，计算cost方式不变，反向传播函数改变

        # 更新参数
        parameters = update_parameters(parameters, grads, learning_rate)

        # 记录并打印成本
        if i % 1000 == 0:
            # 记录成本
            costs.append(cost)
            if print_cost and i % 10000 == 0:
                # 打印成本
                print("第" + str(i) + "次迭代，成本值为：" + str(cost))

    # 是否绘制成本曲线图
    if is_plot:
        plt.plot(costs)
        plt.ylabel('cost')
        plt.xlabel('iterations (x1,000)')
        plt.title("Learning rate =" + str(learning_rate))

    # 返回学习后的参数
    return parameters


plt.figure(1)
# parameters = model(train_X, train_Y, lambd=0.7, learning_rate=0.05)
parameters = model(train_X, train_Y, keep_prob=0.86, learning_rate=0.05)
print("On the train set:")
predictions_train = predict(train_X, train_Y, parameters)
print("On the test set:")
predictions_test = predict(test_X, test_Y, parameters)

plt.figure(2)
plt.title("Model with L2-regularization")
axes = plt.gca()
axes.set_xlim([-0.75, 0.40])
axes.set_ylim([-0.75, 0.65])
plot_decision_boundary(lambda x: predict_dec(parameters, x.T), train_X, train_Y)

plt.show()
