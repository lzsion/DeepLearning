# Author: LZS
# CreateTime: 2023/10/14  21:00
# FileName: Net
# Python Script

# 搭建网络

import torch.nn as nn
import torch.nn.functional as F


class ConvNet(nn.Module):

    def __init__(self):
        super(ConvNet, self).__init__()
        self.for_ward = nn.Sequential(

            nn.Conv2d(1, 16, 5),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(16, 32, 7),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Dropout2d(p=0.5),
            nn.Conv2d(32, 64, 5),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.Dropout2d(p=0.3),
            nn.Conv2d(64, 10, 7),
            nn.BatchNorm2d(10),
        )

    def forward(self, x):
        x = self.for_ward(x)
        x = x.view(x.shape[0], -1)
        x = F.softmax(x, 1)
        return x
