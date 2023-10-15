# Author: LZS
# CreateTime: 2023/10/15  14:53
# FileName: CNN_Net
# Python Script

# 用于mnist分类的CNN网络

import sys
from datetime import datetime
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
from matplotlib import pyplot as plt
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from torch.utils.data import DataLoader, TensorDataset
from plot_func import plot_samples, plot_curve
from load_data import load_data


class ANN(nn.Module):

    def __init__(self):
        super(ANN, self).__init__()
        self.model_linear = nn.Sequential(
            nn.Flatten(),

            nn.Linear(28*28, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),

            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),

            nn.Linear(256, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),

            nn.Linear(64, 10),

        )

    def forward(self, x):
        x = self.model_linear(x)
        x = F.softmax(x, 1)
        return x


