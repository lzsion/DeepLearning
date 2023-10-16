# 深度学习课程笔记 & 代码作业

## 0-相关链接

[吴恩达深度学习第一课](https://www.bilibili.com/video/BV164411m79z/)

[吴恩达深度学习第二课](https://www.bilibili.com/video/BV1V441127zE/)

[吴恩达深度学习第三课](https://www.bilibili.com/video/BV1f4411C7Nx/)

[吴恩达深度学习第四课](https://www.bilibili.com/video/BV1F4411y7o7/)

[吴恩达深度学习第五课](https://www.bilibili.com/video/BV1F4411y7BA/)

[吴恩达深度学习作业链接1](https://zhuanlan.zhihu.com/p/95510114)

[吴恩达深度学习作业链接2](https://zhuanlan.zhihu.com/p/354386182)

[李宏毅机器学习课程](https://www.bilibili.com/video/BV1Wv411h7kN/)

## 1-学习任务要求

### 1.1 神经网络与深度学习

- （1）视频教程：学习吴恩达deeplearning.ai系列教程中第一课所有内容。
    [吴恩达深度学习第一课](https://www.bilibili.com/video/BV164411m79z/)

- （2）学习目标：学习Python和numpy科学计算库的基本用法，了解人工神经网络（ANN）基本组成结构，重点掌握如何利用反向传播算法训练ANN。

- （3）动手实验：完成第一课对应的课后编程作业并撰写报告，报告主要记录实验原理（反向传播算法推导等）、实验环境、实验结果、结果分析等。

### 1.2 网络正则化以及优化方法

- （1）视频教程：学习吴恩达deeplearning.ai系列教程中第二课和第三课所有内容。
    [吴恩达深度学习第二课](https://www.bilibili.com/video/BV1V441127zE/)
    [吴恩达深度学习第三课](https://www.bilibili.com/video/BV1f4411C7Nx/)

- （2）学习目标：了解正则化方法、优化方法、数据集划分方式，学习超参数调节技巧。重点掌握mini-batch梯度下降法和batch norm正则化方法。

- （3）动手实验：完成第二课对应的课后编程作业并撰写报告，报告主要记录实验原理（mini-batch、batch norm理论推导等）、实验环境、实验结果、结果分析等。

### 1.3 卷积神经网络（CNN）

- （1）视频教程：学习吴恩达deeplearning.ai系列教程中第四课前两周内容。
    [吴恩达深度学习第四课](https://www.bilibili.com/video/BV1F4411y7o7/)

- （2）学习目标：学习卷积神经网络的基本结构，了解常见的卷积神经网络。重点掌握卷积和池化的计算以及CNN中的反向传播算法。

- （3）动手实验：完成第四课对应的课后编程作业并撰写报告，报告主要记录实验原理（CNN模型结构、反向传播理论推导等）、实验环境、实验结果、结果分析等。

## 2-分类实战

### 2.1 图像分类实战

- （1）参考资料：demo_LeNet文件和demo_fully_ConvNet文件。
    demo_LeNet和demo_fully_ConvNet的实现代码见  
    /exercise/image-classification/assignment-1/code  
    /exercise/image-classification/assignment-2/code  
- （2）学习目标：学习如何利用pytorch搭建并训练分类网络，包括数据集加载、预处理、网络构建、优化器选择等。重点掌握pytorch的常用函数和语法。
- （3）动手实验：完成作业并撰写报告，作业共两部分内容
    **任务一**为利用pytorch构建用于mnist分类的ANN和CNN网络（网络需包含dropout层）。
    代码见 /exercise/image-classification/assignment-3/code
    **任务二**为利用pytorch构建用于mstar分类的全卷积网络（全卷积指不包含全连接层，此外网络需包含batch norm层）。
    代码见 /exercise/image-classification/assignment-4/code
报告主要记录模型原理（模型结构、前向反向传播、优化器等）、实验环境、数据集介绍、实验结果（loss和accuracy曲线等）、结果对比分析等。

### 2.2 数据集

由于GitHub大小限制，隐藏了部分数据集
/exercise/image-classification/assignment-1/code/data/
/exercise/image-classification/assignment-2/code/mstar_data/
/exercise/image-classification/assignment-3/code/data/
/exercise/image-classification/assignment-4/code/mstar_data/
