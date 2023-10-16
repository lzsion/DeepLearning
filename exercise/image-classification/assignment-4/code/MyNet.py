# Author: LZS
# CreateTime: 2023/10/14  21:00
# FileName: Net
# Python Script

# 搭建网络

import torch.nn as nn
import torch.nn.functional as F


class ConvNet88(nn.Module):

    def __init__(self):
        super(ConvNet88, self).__init__()
        self.for_ward = nn.Sequential(

            nn.Conv2d(1, 16, 5),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(16, 32, 5),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            # nn.Dropout2d(p=0.5),
            nn.Conv2d(32, 64, 6),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            # nn.Dropout2d(p=0.3),
            nn.Conv2d(64, 128, 5),
            nn.BatchNorm2d(128),
            nn.ReLU(),

            nn.Conv2d(128, 10, 3),
            nn.BatchNorm2d(10),
        )

    def forward(self, x):
        x = self.for_ward(x)
        x = x.view(x.shape[0], -1)
        x = F.softmax(x, 1)
        return x


class ConvNet128(nn.Module):

    def __init__(self):
        super(ConvNet128, self).__init__()
        self.for_ward = nn.Sequential(

            nn.Conv2d(1, 16, 5),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(16, 32, 5),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Dropout2d(p=0.25),
            nn.Conv2d(32, 64, 6),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            # nn.Dropout2d(p=0.3),
            nn.Conv2d(64, 128, 5),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(128, 10, 4),
            nn.BatchNorm2d(10),
        )

    def forward(self, x):
        x = self.for_ward(x)
        x = x.view(x.shape[0], -1)
        x = F.softmax(x, 1)
        return x
