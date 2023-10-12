# Author: LZS
# CreateTime: 2023/10/11  19:20
# FileName: C02W03_1
# Python Script

# tf用法示例

import math
import numpy as np
import h5py
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.python.framework import ops

import tf_utils
import time


# a = tf.constant(2)
# b = tf.constant(10)
# c = tf.multiply(a, b)
#
# # print(c)
# print(c.numpy())


# # 占位符
# x = tf.Variable(0, dtype=tf.int64, name="x")
# x.assign(4)  # 赋值 x 为 4
# result = 2 * x
# print(result.numpy())

def linear_function():
    """
    实现一个线性功能：
        初始化W，类型为tensor的随机变量，维度为(4,3)
        初始化X，类型为tensor的随机变量，维度为(3,1)
        初始化b，类型为tensor的随机变量，维度为(4,1)
    返回：
        result - 运行了session后的结果，运行的是Y = WX + b
    """

    np.random.seed(1)  # 指定随机种子

    X = np.random.randn(3, 1)
    W = np.random.randn(4, 3)
    b = np.random.randn(4, 1)

    Y = tf.add(tf.matmul(W, X), b)  # tf.matmul是矩阵乘法
    # Y = tf.matmul(W,X) + b #也可以以写成这样子

    return Y.numpy()


# print(linear_function())

def sigmoid(z):
    """
    实现使用sigmoid函数计算z
    参数：
        z - 输入的值，标量或矢量
    返回：
        result - 用sigmoid计算z的值
    """

    # 创建一个 TensorFlow 变量，不需要指定数据类型
    x = tf.Variable(z, dtype=tf.float32, name="x")

    # 计算 sigmoid(z)
    sigmoid = tf.sigmoid(x)

    # 不再需要创建会话，可以直接使用 .numpy() 方法获取结果
    result = sigmoid.numpy()

    return result


# print(sigmoid(0))


def one_hot_matrix(labels, C):
    """
    创建一个矩阵，其中第i行对应第i个类号，第j列对应第j个训练样本
    所以如果第j个样本对应着第i个标签，那么entry (i,j)将会是1
    参数：
        labels - 标签向量
        C - 分类数
    返回：
        one_hot - 独热矩阵
    """

    # 使用 tf.one_hot 来创建独热矩阵
    one_hot_matrix = tf.one_hot(labels, depth=C, axis=0, dtype=tf.float32)

    return one_hot_matrix


# one_hot = one_hot_matrix([1, 2, 0, 3], 4)
# print(one_hot)


# def cost(logits, labels):
#     """
#     计算cost
#     参数:
#         logits -- 最后sigmoid前的z的向量，形状 (m,)
#         labels -- y的向量 (1 or 0)，形状 (m,)
#     (在tf中logits对应z，labels对应y)
#     返回:
#         cost -- 每个样本的损失向量，形状 (m,)
#     """
#
#     # 将 labels 转换为 float32 类型
#     labels = tf.cast(labels, dtype=tf.float32)
#
#     # 使用 tf.keras.losses.binary_crossentropy 来计算每个样本的交叉熵损失
#     cost = tf.keras.losses.binary_crossentropy(logits, labels, from_logits=False)
#
#     return cost
#
#

def cost(logits, labels):
    """
    计算cost
    参数:
        logits -- 最后sigmoid前的z的向量，形状 (m,)
        labels -- y的向量 (1 or 0)，形状 (m,)
    (在tf中logits对应z，labels对应y)
    返回:
        cost -- 每个样本的损失向量，形状 (m,)
    """

    z = tf.constant(logits, dtype=tf.float32, name="z")
    y = tf.constant(labels, dtype=tf.float32, name="y")

    # Calculate binary cross entropy cost
    cost = tf.keras.losses.binary_crossentropy(y, z)

    return cost.numpy()


# logits = sigmoid(np.array([0.2, 0.4, 0.7, 0.9], dtype=np.float32).reshape(4, 1))
# labels = np.array([0, 0, 1, 1], dtype=np.float32).reshape(4, 1)
# cost_value = cost(logits, labels)
#
# print("cost = " + str(cost_value))


