#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------- 原 Tiny-ImageNet 模型（保留） ----------------------
class BasicBlock(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(out_ch)
        self.down  = None
        if stride != 1 or in_ch != out_ch:
            self.down = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_ch)
            )

    def forward(self, x):
        identity = x
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = self.bn2(self.conv2(out))
        if self.down is not None:
            identity = self.down(identity)
        out = F.relu(out + identity, inplace=True)
        return out

class CNNTinyImageNet(nn.Module):
    def __init__(self, num_classes=200):
        super().__init__()
        self.num_classes = num_classes
        self.stem = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.layer1 = nn.Sequential(
            BasicBlock(64, 64, stride=1),
            BasicBlock(64, 64, stride=1),
        )
        self.layer2 = nn.Sequential(
            BasicBlock(64, 128, stride=2),
            BasicBlock(128, 128, stride=1),
        )
        self.layer3 = nn.Sequential(
            BasicBlock(128, 256, stride=2),
            BasicBlock(256, 256, stride=1),
        )
        self.layer4 = nn.Sequential(
            BasicBlock(256, 512, stride=2),
            BasicBlock(512, 512, stride=1),
        )
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d((1,1)),
            nn.Flatten(),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.head(x)
        return x
# ----------------------------------------------------------------------


# ------------------------- 新增：SUSY 的 1D-CNN -------------------------
class CNNSUSY(nn.Module):
    """
    适配 SUSY（18 维数值特征，二分类）的 1D-CNN：
    - 先把 [B, 18] 视作 1 个通道的长度 18 的序列 -> [B, 1, 18]
    - 两层 1D 卷积 + BN + ReLU，做轻量局部交互
    - GAP 汇聚后接两层 MLP 分类
    - 参数量小，易于在 FL 中稳定收敛
    """
    def __init__(self, num_classes=2, input_dim=18):
        super().__init__()
        self.input_dim = input_dim
        self.features = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.Conv1d(32, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool1d(1)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),           # [B, 64, 1] -> [B, 64]
            nn.Linear(64, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        # x: [B, D] or [B, 1, D]
        if x.dim() == 2:
            x = x.unsqueeze(1)      # -> [B, 1, D]
        x = self.features(x)        # -> [B, 64, 1]
        x = self.classifier(x)      # -> [B, C]
        return x
# ----------------------------------------------------------------------
