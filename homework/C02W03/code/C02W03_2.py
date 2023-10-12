# Author: LZS
# CreateTime: 2023/10/12  13:04
# FileName: C02W03_2
# Python Script

# tf构建网络

import math
import numpy as np
import h5py
import matplotlib.pyplot as plt
import tensorflow as tf
# from tensorflow.python.framework import ops

import tf_utils
import time

# 加载数据集
X_train_orig, Y_train_orig, X_test_orig, Y_test_orig, classes = tf_utils.load_dataset()

# index = 15
# plt.imshow(X_train_orig[index])
# print(X_train_orig.shape)
# print(Y_train_orig.shape)
# print("Y = " + str(np.squeeze(Y_train_orig[:, index])))
# plt.show()

# 扁平化数据
# 每一列就是一个样本
X_train_flatten = X_train_orig.reshape(X_train_orig.shape[0], -1).T
X_test_flatten = X_test_orig.reshape(X_test_orig.shape[0], -1).T

# 归一化数据
X_train = X_train_flatten / 255
X_test = X_test_flatten / 255

# 转换为独热矩阵
Y_train = tf_utils.convert_to_one_hot(Y_train_orig, 6)
Y_test = tf_utils.convert_to_one_hot(Y_test_orig, 6)


# print("训练集样本数 = " + str(X_train.shape[1]))
# print("测试集样本数 = " + str(X_test.shape[1]))
# print("X_train.shape: " + str(X_train.shape))
# print("Y_train.shape: " + str(Y_train.shape))
# print("X_test.shape: " + str(X_test.shape))
# print("Y_test.shape: " + str(Y_test.shape))

# 训练集样本数 = 1080
# 测试集样本数 = 120
# X_train.shape: (12288, 1080)
# Y_train.shape: (6, 1080)
# X_test.shape: (12288, 120)
# Y_test.shape: (6, 120)

def create_placeholders(n_x, n_y):
    """
    为TensorFlow会话创建占位符
    参数：
        n_x - 一个实数，图片向量的大小（64*64*3 = 12288）
        n_y - 一个实数，分类数（从0到5，所以n_y = 6）
    返回：
        X - 一个数据输入的占位符，维度为[n_x, None]，dtype = "float"
        Y - 一个对应输入的标签的占位符，维度为[n_Y, None]，dtype = "float"

    提示：
        使用None，因为它让我们可以灵活处理占位符提供的样本数量。事实上，测试/训练期间的样本数量是不同的。
    """
    X = tf.keras.layers.Input(shape=(n_x, None), dtype=tf.float32, name="X")
    Y = tf.keras.layers.Input(shape=(n_y, None), dtype=tf.float32, name="Y")

    return X, Y


def initialize_parameters():
    """
    初始化神经网络的参数，参数的维度如下：
        W1 : [25, 12288]
        b1 : [25, 1]
        W2 : [12, 25]
        b2 : [12, 1]
        W3 : [6, 12]
        b3 : [6, 1]
    返回：
        parameters - 包含了W和b的字典
    """

    tf.random.set_seed(1)  # 指定随机种子

    initializer = tf.initializers.GlorotUniform(seed=1)  # 初始化方式

    W1 = tf.Variable(initializer(shape=(25, 12288), dtype=tf.float32), name="W1")
    b1 = tf.Variable(tf.zeros((25, 1), dtype=tf.float32), name="b1")
    W2 = tf.Variable(initializer(shape=(12, 25), dtype=tf.float32), name="W2")
    b2 = tf.Variable(tf.zeros((12, 1), dtype=tf.float32), name="b2")
    W3 = tf.Variable(initializer(shape=(6, 12), dtype=tf.float32), name="W3")
    b3 = tf.Variable(tf.zeros((6, 1), dtype=tf.float32), name="b3")

    parameters = {
        "W1": W1,
        "b1": b1,
        "W2": W2,
        "b2": b2,
        "W3": W3,
        "b3": b3
    }

    return parameters


def forward_propagation(X, parameters):
    """
    实现一个模型的前向传播，模型结构为LINEAR -> RELU -> LINEAR -> RELU -> LINEAR -> SOFTMAX
    参数：
        X - 输入数据，维度为（输入节点数量，样本数量）
        parameters - 包含了W和b的参数的字典
    返回：
        Z3 - 最后一个LINEAR节点的输出
    """

    W1 = parameters['W1']
    b1 = parameters['b1']
    W2 = parameters['W2']
    b2 = parameters['b2']
    W3 = parameters['W3']
    b3 = parameters['b3']

    Z1 = tf.add(tf.matmul(W1, X), b1)  # Z1 = np.dot(W1, X) + b1
    A1 = tf.nn.relu(Z1)  # A1 = relu(Z1)
    Z2 = tf.add(tf.matmul(W2, A1), b2)  # Z2 = np.dot(W2, a1) + b2
    A2 = tf.nn.relu(Z2)  # A2 = relu(Z2)
    Z3 = tf.add(tf.matmul(W3, A2), b3)  # Z3 = np.dot(W3, A2) + b3

    return Z3


def compute_cost(Z3, Y):
    """
    计算成本
    参数：
        Z3 - 前向传播的结果
        Y - 标签，一个张量，与Z3的形状相同
    返回：
        cost - 成本值
    """

    logits = tf.transpose(Z3)  # 转置
    labels = tf.transpose(Y)

    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels))

    return cost


# def model(X_train, Y_train, X_test, Y_test,
#           learning_rate=0.0001, num_epochs=1500, minibatch_size=32,
#           print_cost=True, is_plot=True):
#     """
#     实现一个三层的TensorFlow神经网络：LINEAR -> RELU -> LINEAR -> RELU -> LINEAR -> SOFTMAX
#     参数：
#         X_train - 训练集，维度为（样本数量, 输入节点数量）
#         Y_train - 训练集分类数量，维度为（样本数量, 输出节点数量）
#         X_test - 测试集，维度为（样本数量, 输入节点数量）
#         Y_test - 测试集分类数量，维度为（样本数量, 输出节点数量）
#         learning_rate - 学习速率
#         num_epochs - 整个训练集的遍历次数
#         minibatch_size - 每个小批量数据集的大小
#         print_cost - 是否打印成本，每100代打印一次
#         is_plot - 是否绘制曲线图
#     返回：
#         parameters - 学习后的参数
#     """
#
#     tf.random.set_seed(1)
#     m = X_train.shape[1]  # 样本数量
#     n_x = X_train.shape[0]  # 输入节点数量
#     n_y = Y_train.shape[0]  # 输出节点数量
#     costs = []  # 成本集
#
#     # 创建模型
#     model = tf.keras.Sequential([
#         tf.keras.layers.Input(shape=(n_x,)),
#         tf.keras.layers.Dense(25, activation='relu', kernel_initializer='glorot_uniform'),
#         tf.keras.layers.Dense(12, activation='relu', kernel_initializer='glorot_uniform'),
#         tf.keras.layers.Dense(n_y, activation='softmax')
#     ])
#
#     # 编译模型
#     model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
#                   loss='categorical_crossentropy', metrics=['accuracy'])
#
#     # # 初始化参数
#     # parameters = {
#     #     'W1': model.layers[0].get_weights()[0],
#     #     'b1': model.layers[0].get_weights()[1],
#     #     'W2': model.layers[1].get_weights()[0],
#     #     'b2': model.layers[1].get_weights()[1],
#     #     'W3': model.layers[2].get_weights()[0],
#     #     'b3': model.layers[2].get_weights()[1],
#     # }
#
#     for epoch in range(num_epochs):
#         # 随机划分
#         minibatches = tf_utils.random_mini_batches(X_train, Y_train, minibatch_size)
#
#         for minibatch in minibatches:
#             # 每个minibatch进行一次更新参数
#             (minibatch_X, minibatch_Y) = minibatch
#             minibatch_X = tf.transpose(minibatch_X)
#             minibatch_Y = tf.transpose(minibatch_Y)
#             # print(minibatch_X.shape)
#             # print(minibatch_Y.shape)
#             model.fit(minibatch_X, minibatch_Y, epochs=1, verbose=0)
#
#         # 一次epoch完成后 计算成本
#         minibatch_cost = model.evaluate(minibatch_X, minibatch_Y, verbose=0)[0]
#         epoch_cost = minibatch_cost / len(minibatches)
#
#         if epoch % 100 == 0 and print_cost:
#             print(f"Epoch {epoch}: cost = {epoch_cost}")
#
#         costs.append(epoch_cost)
#
#     # 计算准确率
#     train_loss, train_accuracy = model.evaluate(X_train, Y_train)
#     test_loss, test_accuracy = model.evaluate(X_test, Y_test)
#
#     print("训练集的准确率:", train_accuracy)
#     print("测试集的准确率:", test_accuracy)
#
#     if is_plot:
#         plt.plot(np.squeeze(costs))
#         plt.ylabel('cost')
#         plt.xlabel('epochs')
#         plt.title("Learning rate =" + str(learning_rate))
#         plt.show()
#
#     # return parameters
#
#
# # print(X_train.shape)  # (12288, 1080)
# # print(Y_train.shape)  # (6, 1080)
# #
# # print(X_test.shape)  # (12288, 120)
# # print(Y_test.shape)  # (6, 120)
#
# model(X_train, Y_train, X_test, Y_test)
#
# start_time = time.perf_counter()
# # 开始训练
# # parameters = model(X_train, Y_train, X_test, Y_test)
# model(X_train, Y_train, X_test, Y_test)
# # 结束时间
# end_time = time.perf_counter()
# # 计算时差
# print("CPU的执行时间 = " + str(end_time - start_time) + " 秒")
def model(X_train, Y_train, X_test, Y_test,
          learning_rate=0.0001, num_epochs=1500, minibatch_size=128,
          print_cost=True, is_plot=True):
    """
    实现一个三层的TensorFlow神经网络：LINEAR -> RELU -> LINEAR -> RELU -> LINEAR -> SOFTMAX
    参数：
        X_train - 训练集，维度为（样本数量, 输入节点数量）
        Y_train - 训练集分类数量，维度为（样本数量, 输出节点数量）
        X_test - 测试集，维度为（样本数量, 输入节点数量）
        Y_test - 测试集分类数量，维度为（样本数量, 输出节点数量）
        learning_rate - 学习速率
        num_epochs - 整个训练集的遍历次数
        minibatch_size - 每个小批量数据集的大小
        print_cost - 是否打印成本，每100代打印一次
        is_plot - 是否绘制曲线图
    返回：
        parameters - 学习后的参数
    """

    tf.random.set_seed(1)
    m = X_train.shape[1]  # 样本数量
    n_x = X_train.shape[0]  # 输入节点数量
    n_y = Y_train.shape[0]  # 输出节点数量
    costs = []  # 成本集

    # 创建模型
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(n_x, )),
        tf.keras.layers.Dense(25, activation='relu', kernel_initializer='glorot_uniform'),
        tf.keras.layers.Dense(12, activation='relu', kernel_initializer='glorot_uniform'),
        tf.keras.layers.Dense(n_y, activation='softmax')
    ])

    # 编译模型
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
                  loss='categorical_crossentropy', metrics=['accuracy'])

    for epoch in range(num_epochs):
        # 随机划分
        minibatches = tf_utils.random_mini_batches(X_train, Y_train, minibatch_size)

        for minibatch in minibatches:
            # 每个minibatch进行一次更新参数
            (minibatch_X, minibatch_Y) = minibatch
            # minibatch_X = tf.transpose(minibatch_X)
            # minibatch_Y = tf.transpose(minibatch_Y)
            model.fit(minibatch_X.T, minibatch_Y.T, epochs=1, verbose=0)

        # 一次epoch完成后 计算成本
        minibatch_cost = model.evaluate(minibatch_X.T, minibatch_Y.T, verbose=0)[0]
        epoch_cost = minibatch_cost / len(minibatches)

        if epoch % 10 == 0 and print_cost:
            print(f"Epoch {epoch}: cost = {epoch_cost}")

        costs.append(epoch_cost)

    # 计算准确率
    train_loss, train_accuracy = model.evaluate(X_train.T, Y_train.T)
    test_loss, test_accuracy = model.evaluate(X_test.T, Y_test.T)

    print("训练集的准确率:", train_accuracy)
    print("测试集的准确率:", test_accuracy)

    if is_plot:
        plt.plot(np.squeeze(costs))
        plt.ylabel('cost')
        plt.xlabel('epochs')
        plt.title("Learning rate =" + str(learning_rate))
        plt.show()

    # return parameters


# print(X_train.shape)  # (12288, 1080)
# print(Y_train.shape)  # (6, 1080)
#
# print(X_test.shape)  # (12288, 120)
# print(Y_test.shape)  # (6, 120)

with tf.device('/gpu:0'):
    model(X_train, Y_train, X_test, Y_test)

# start_time = time.perf_counter()
# # 开始训练
# # parameters = model(X_train, Y_train, X_test, Y_test)
# model(X_train, Y_train, X_test, Y_test)
# # 结束时间
# end_time = time.perf_counter()
# # 计算时差
# print("CPU的执行时间 = " + str(end_time - start_time) + " 秒")

