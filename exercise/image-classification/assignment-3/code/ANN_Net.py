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


class CNN(nn.Module):

    def __init__(self):
        super(CNN, self).__init__()
        self.model_conv = nn.Sequential(
            nn.Conv2d(1, 16, 5, 1, 0),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            # nn.Dropout2d(p=0.8),
            nn.Conv2d(16, 32, 5, 1, 0),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
        )
        self.model_linear = nn.Sequential(
            nn.Linear(32 * 4 * 4, 10),
            # nn.ReLU(),
        )

    def forward(self, x):
        x = self.model_conv(x)
        x = x.view(x.shape[0], -1)
        x = self.model_linear(x)
        x = F.softmax(x, 1)
        return x


