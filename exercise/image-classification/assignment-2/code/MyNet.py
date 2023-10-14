# Author: LZS
# CreateTime: 2023/10/14  21:00
# FileName: Net
# Python Script

# 搭建网络

import torch.nn as nn
import torch.nn.functional as F

class Convnet(nn.Module):

    def __init__(self):
        super(Convnet, self).__init__()
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


def forward_hook0(module, inp, outp):
    fmap_block0['input'] = inp
    fmap_block0['output'] = outp


def forward_hook1(module, inp, outp):
    fmap_block1['input'] = inp
    fmap_block1['output'] = outp


def forward_hook2(module, inp, outp):
    fmap_block2['input'] = inp
    fmap_block2['output'] = outp
