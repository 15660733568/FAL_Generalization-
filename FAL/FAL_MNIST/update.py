#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from models import PGD_Linf_Attack


class DatasetSplit(Dataset):
    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = [int(i) for i in idxs]

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        return image.clone(), torch.tensor(label)


class LocalUpdate(object):
    def __init__(self, args, dataset, idxs, logger, epsilon=0.1):
        self.args = args
        self.logger = logger
        self.trainloader, self.validloader, self.testloader = self.train_val_test(dataset, list(idxs))
        self.device = 'cuda' if args.gpu else 'cpu'
        self.criterion = nn.NLLLoss().to(self.device)
        self.epsilon = epsilon
        self.pgd_steps = args.pgd_steps  # 固定攻击迭代次数

    def train_val_test(self, dataset, idxs):
        idxs_train = idxs[:int(0.8 * len(idxs))]
        idxs_val = idxs[int(0.8 * len(idxs)):int(0.9 * len(idxs))]
        idxs_test = idxs[int(0.9 * len(idxs)):]

        trainloader = DataLoader(DatasetSplit(dataset, idxs_train),
                               batch_size=self.args.local_bs, shuffle=True)
        validloader = DataLoader(DatasetSplit(dataset, idxs_val),
                               batch_size=64, shuffle=False)
        testloader = DataLoader(DatasetSplit(dataset, idxs_test),
                              batch_size=64, shuffle=False)
        return trainloader, validloader, testloader

    def update_weights(self, model, global_round, current_lr=None):
        model.train()
        optimizer = torch.optim.SGD(model.parameters(),
                                    lr=current_lr or self.args.lr,
                                    momentum=self.args.momentum)

        # 固定攻击迭代次数，不随epsilon变化
        pgd_attack = PGD_Linf_Attack(model,
                                     epsilon=self.epsilon,
                                     alpha=0.01,  # 固定步长
                                     steps=self.pgd_steps,  # 固定迭代次数
                                     random_start=True)

        for _ in range(self.args.local_ep):
            for images, labels in self.trainloader:
                images, labels = images.to(self.device), labels.to(self.device)

                model.zero_grad()

                # 干净样本损失
                outputs = model(images)
                clean_loss = self.criterion(outputs, labels)

                # epsilon=0时完全使用干净样本训练
                if self.epsilon == 0.0:
                    clean_loss.backward()
                    optimizer.step()
                    continue

                # 生成对抗样本
                adv_images = pgd_attack.perturb(images, labels)
                adv_outputs = model(adv_images)
                adv_loss = self.criterion(adv_outputs, labels)

                # 动态调整损失权重: epsilon越大，对抗损失权重越大
                # 使用sigmoid函数使权重变化更平滑
                adv_weight = 2 * torch.sigmoid(torch.tensor(5.0 * self.epsilon)).item()
                clean_weight = 1.0

                total_loss = clean_weight * clean_loss + adv_weight * adv_loss
                total_loss.backward()
                optimizer.step()

        return model.state_dict(), total_loss.item() if self.epsilon != 0.0 else clean_loss.item()

    def inference(self, model):
        model.eval()
        loss, total, correct = 0.0, 0.0, 0.0

        for batch_idx, (images, labels) in enumerate(self.testloader):
            images, labels = images.to(self.device), labels.to(self.device)

            outputs = model(images)
            batch_loss = self.criterion(outputs, labels)
            loss += batch_loss.item()

            _, pred_labels = torch.max(outputs, 1)
            pred_labels = pred_labels.view(-1)
            correct += torch.sum(torch.eq(pred_labels, labels)).item()
            total += len(labels)

        accuracy = correct/total
        return accuracy, loss


def test_inference(args, model, test_dataset):
    model.eval()
    loss, total, correct = 0.0, 0.0, 0.0

    device = 'cuda' if args.gpu else 'cpu'
    criterion = nn.NLLLoss().to(device)
    testloader = DataLoader(test_dataset, batch_size=128,
                          shuffle=False)

    for batch_idx, (images, labels) in enumerate(testloader):
        images, labels = images.to(device), labels.to(device)

        outputs = model(images)
        batch_loss = criterion(outputs, labels)
        loss += batch_loss.item()

        _, pred_labels = torch.max(outputs, 1)
        pred_labels = pred_labels.view(-1)
        correct += torch.sum(torch.eq(pred_labels, labels)).item()
        total += len(labels)

    accuracy = correct/total
    return accuracy, loss


def test_adv_inference(args, model, test_dataset, epsilon_list=[0.0, 0.1, 0.3, 0.5]):
    """测试模型在不同epsilon下的对抗鲁棒性
    Args:
        args: 命令行参数
        model: 要测试的模型
        test_dataset: 测试数据集
        epsilon_list: 要测试的epsilon值列表

    Returns:
        dict: 包含每个epsilon下的测试结果，格式为:
            {
                'epsilon_0.0': {'accuracy': 0.98, 'loss': 0.05},
                'epsilon_0.1': {'accuracy': 0.85, 'loss': 0.32},
                ...
            }
    """
    model.eval()  # 设置为评估模式
    device = 'cuda' if args.gpu else 'cpu'
    criterion = nn.NLLLoss().to(device)  # 使用负对数似然损失
    testloader = DataLoader(test_dataset, batch_size=128, shuffle=False)

    results = {}  # 存储所有epsilon的结果

    for epsilon in epsilon_list:
        # 初始化PGD攻击器
        pgd_attack = PGD_Linf_Attack(
            model=model,
            epsilon=epsilon,
            alpha=0.01,  # 攻击步长
            steps=7  # PGD迭代次数
        )

        total_loss = 0.0
        total, correct = 0.0, 0.0

        for batch_idx, (images, labels) in enumerate(testloader):
            images, labels = images.to(device), labels.to(device)

            # 生成对抗样本
            adv_images = pgd_attack.perturb(images, labels)

            # 在对抗样本上测试
            with torch.no_grad():  # 禁用梯度计算
                outputs = model(adv_images)
                loss = criterion(outputs, labels)

            # 累计统计量
            total_loss += loss.item()
            _, pred_labels = torch.max(outputs, 1)
            correct += torch.sum(torch.eq(pred_labels, labels)).item()
            total += len(labels)

        # 计算并存储当前epsilon的结果
        accuracy = correct / total
        avg_loss = total_loss / len(testloader)

        results[f'epsilon_{epsilon}'] = {
            'accuracy': accuracy,
            'loss': avg_loss
        }

        # 打印当前epsilon的结果
        print(f'ε={epsilon:.1f} | Test Acc: {accuracy:.2%} | Loss: {avg_loss:.4f}')

    return results


class LRScheduler:
    def __init__(self, optimizer, args):
        self.optimizer = optimizer
        self.args = args
        self.current_lr = args.lr
        self.global_round = 0
        self.last_decay_round = 0

        # 预热参数
        self.warmup_epochs = 5  # 预热轮数
        self.warmup_lr = args.lr * 0.1  # 初始学习率

    def step(self):
        """Update learning rate based on decay strategy"""
        self.global_round += 1

        # 预热阶段
        if self.global_round <= self.warmup_epochs:
            self.current_lr = self.warmup_lr + (self.args.lr - self.warmup_lr) * \
                              (self.global_round / self.warmup_epochs)
        else:
            # 正常衰减阶段
            if self.args.lr_decay == 'step':
                if self.global_round - self.last_decay_round >= self.args.lr_decay_step:
                    self.current_lr = max(self.current_lr * self.args.lr_decay_rate,
                                          self.args.min_lr)
                    self.last_decay_round = self.global_round
            elif self.args.lr_decay == 'exp':
                decay_rate = (self.global_round - self.warmup_epochs) / self.args.lr_decay_step
                self.current_lr = max(
                    self.args.lr * (self.args.lr_decay_rate ** decay_rate),
                    self.args.min_lr)
            elif self.args.lr_decay == 'cosine':
                import math
                progress = (self.global_round - self.warmup_epochs) / (self.args.epochs - self.warmup_epochs)
                cosine_decay = 0.5 * (1 + math.cos(math.pi * progress))
                self.current_lr = max(self.args.min_lr + (self.args.lr - self.args.min_lr) * cosine_decay,
                                      self.args.min_lr)

        # 更新优化器的学习率
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.current_lr

        return self.current_lr