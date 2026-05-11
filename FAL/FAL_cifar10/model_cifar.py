
# @Time   : 2022/8/25 14:43
# @Author :lgl
# @e-mail :GuanlinLi_BIT@163.com


import torch
from torch.nn import Module, Sequential, Conv2d, MaxPool2d, Flatten, Linear, ReLU, BatchNorm2d, Dropout
from torch.utils.data import DataLoader

class CNNCifar(Module):
    def __init__(self):
        super(CNNCifar, self).__init__()
        self.neuralnet = Sequential(
            Conv2d(in_channels=3, out_channels=64, kernel_size=(3, 3), padding=1),  # 1卷积层
            BatchNorm2d(64),
            Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), padding=1),  # 1卷积层
            BatchNorm2d(64),
            ReLU(inplace=True),

            MaxPool2d(kernel_size=2, ceil_mode=False),  # 1最大池化层

            Conv2d(in_channels=64, out_channels=128, kernel_size=(3, 3), padding=1),  # 1卷积层
            BatchNorm2d(128),
            Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), padding=1),  # 1卷积层
            BatchNorm2d(128),
            ReLU(inplace=True),
            MaxPool2d(kernel_size=2, ceil_mode=False),  # 1最大池化层

            Conv2d(in_channels=128, out_channels=256, kernel_size=(3, 3), padding=1),  # 1卷积层
            BatchNorm2d(256),
            Conv2d(in_channels=256, out_channels=256, kernel_size=(3, 3), padding=1),  # 1卷积层
            BatchNorm2d(256),
            ReLU(inplace=True),
            MaxPool2d(kernel_size=2, ceil_mode=False),  # 1最大池化层

            Conv2d(in_channels=256, out_channels=512, kernel_size=(3, 3), padding=1),  # 1卷积层
            BatchNorm2d(512),
            Conv2d(in_channels=512, out_channels=512, kernel_size=(3, 3), padding=1),  # 1卷积层
            BatchNorm2d(512),
            ReLU(inplace=True),
            MaxPool2d(kernel_size=2, ceil_mode=False),  # 1最大池化层

            # Conv2d(in_channels=3, out_channels=32, kernel_size=(5, 5), padding=2),  # 1 2卷积层
            # ReLU(inplace=True),
            # MaxPool2d(kernel_size=2, ceil_mode=True),  # 2最大池化层
            #
            # Conv2d(in_channels=32, out_channels=32, kernel_size=(5, 5), padding=2),  # 2 3卷积层
            # ReLU(inplace=True),
            # MaxPool2d(kernel_size=2, ceil_mode=True),  # 3最大池化层
            #
            # Conv2d(in_channels=32, out_channels=64, kernel_size=(5, 5), padding=2),  # 3 5卷积层
            # ReLU(inplace=True),
            # MaxPool2d(kernel_size=2, ceil_mode=True),  # 6最大池化层
            #
            Flatten(),  # 7 Flatten层
            Dropout(0.4),
            Linear(2048, 256),  # 8 全连接层
            Linear(256, 64),  # 8 全连接层
            Linear(64, 10)  # 9 全连接层
        )

    def forward(self, input):
        out = self.neuralnet(input)
        return out


neural_networks = CNNCifar()
