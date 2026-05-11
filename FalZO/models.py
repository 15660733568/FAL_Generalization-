#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

from torch import nn
import torch.nn.functional as F
import torch

class MLP(nn.Module):
    def __init__(self, dim_in, dim_hidden, dim_out):
        super(MLP, self).__init__()
        self.layer_input = nn.Linear(dim_in, dim_hidden)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout()
        self.layer_hidden = nn.Linear(dim_hidden, dim_out)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = x.view(-1, x.shape[1]*x.shape[-2]*x.shape[-1])
        x = self.layer_input(x)
        x = self.dropout(x)
        x = self.relu(x)
        x = self.layer_hidden(x)
        return self.softmax(x)


class CNNMnist(nn.Module):
    def __init__(self, args):
        super(CNNMnist, self).__init__()
        self.conv1 = nn.Conv2d(args.num_channels, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, args.num_classes)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, x.shape[1]*x.shape[2]*x.shape[3])
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


class CNNFashion_Mnist(nn.Module):
    def __init__(self, args):
        super(CNNFashion_Mnist, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=5, padding=2),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=5, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2))
        self.fc = nn.Linear(7 * 7 * 32, 10)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out


class CNNCifar(nn.Module):
    def __init__(self, args):
        super(CNNCifar, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, args.num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return F.log_softmax(x, dim=1)


class PGD_Linf_Attack:
    def __init__(self, model, epsilon=0.3, alpha=0.01, steps=40, random_start=True, zo=True, zo_mu=0.001,
                 zo_num_dir=10):
        self.model = model
        self.epsilon = epsilon
        self.alpha = alpha
        self.steps = steps
        self.random_start = random_start
        self.zo = zo
        self.zo_mu = zo_mu
        self.zo_num_dir = zo_num_dir

    def perturb(self, images, labels):
        images = images.clone().detach()
        labels = labels.clone().detach()

        if self.random_start:
            # 增加初始扰动随机性
            delta = torch.empty_like(images).uniform_(-self.epsilon, self.epsilon)
            images = torch.clamp(images + delta, min=0, max=1)

        if self.zo:  # 零阶PGD攻击 (Zeroth-order)
            for _ in range(self.steps):
                grad_est = torch.zeros_like(images)
                for _ in range(self.zo_num_dir):
                    u = torch.randn_like(images)
                    u /= (torch.norm(u) + 1e-8)
                    f_pos = F.nll_loss(self.model(torch.clamp(images + self.zo_mu * u, 0, 1)), labels)
                    f_neg = F.nll_loss(self.model(torch.clamp(images - self.zo_mu * u, 0, 1)), labels)
                    grad_est += ((f_pos - f_neg) / (2 * self.zo_mu)) * u
                grad_est /= self.zo_num_dir

                # 修改步长：扰动越大，步长越大
                step_size = self.alpha * (1 + self.epsilon * 2)
                images = images + step_size * grad_est.sign()
                eta = torch.clamp(images - images.clone().detach(), min=-self.epsilon, max=self.epsilon)
                images = torch.clamp(images.clone().detach() + eta, min=0, max=1).detach()
        else:  # 一阶PGD攻击 (原始梯度)
            for _ in range(self.steps):
                images.requires_grad = True
                outputs = self.model(images)
                loss = F.nll_loss(outputs, labels)
                self.model.zero_grad()
                loss.backward()

                with torch.no_grad():
                    # 修改步长：扰动越大，步长越大
                    step_size = self.alpha * (1 + self.epsilon * 2)
                    adv_images = images + step_size * images.grad.sign()
                    eta = torch.clamp(adv_images - images, min=-self.epsilon, max=self.epsilon)
                    images = torch.clamp(images + eta, min=0, max=1).detach_()

        return images