# Author: LZS
# CreateTime: 2023/10/10  21:17
# FileName: C02W01_1
# Python Script


import numpy as np
import matplotlib.pyplot as plt
import sklearn
import sklearn.datasets
import init_utils

from testCases import *
from gc_utils import sigmoid, relu, dictionary_to_vector, vector_to_dictionary, gradients_to_vector

plt.rcParams['figure.figsize'] = (7.0, 4.0)  # 设置 plot 的默认大小
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

train_X, train_Y, test_X, test_Y = init_utils.load_dataset()

plt.scatter(train_X[0, :], train_X[1, :], c=train_Y.reshape(train_X[0, :].shape), s=40, cmap=plt.cm.Spectral)
plt.show()
