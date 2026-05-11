#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from models import PGD_Linf_Attack
import copy

# ============ Moreau Envelope核心函数 ============

def prox_term(model, envelope_model):
    prox = 0.0
    for param, param_u in zip(model.parameters(), envelope_model.parameters()):
        prox += torch.sum((param - param_u) ** 2)
    return prox

def step_u(envelope_model, model, beta):
    for param_u, param in zip(envelope_model.parameters(), model.parameters()):
        param_u.data += beta * (param.data - param_u.data)

# ============ 联邦本地数据分割 ============

class DatasetSplit(Dataset):
    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = [int(i) for i in idxs]

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        return image.clone(), torch.tensor(label)

# ============ 联邦本地训练（集成ME-A） ============

class LocalUpdate(object):
    def __init__(self, args, dataset, idxs, logger, epsilon=0.1):
        self.args = args
        self.logger = logger
        self.trainloader, self.validloader, self.testloader = self.train_val_test(dataset, list(idxs))
        self.device = 'cuda' if args.gpu else 'cpu'
        self.criterion = nn.NLLLoss().to(self.device)
        self.epsilon = epsilon
        self.pgd_steps = args.pgd_steps  # 固定攻击迭代次数

        # Moreau Envelope参数（可以被命令行覆盖）
        self.rho = getattr(args, 'me_rho', 5e-4)
        self.me_alpha = getattr(args, 'me_alpha', 0.01)
        self.envelope_model = None

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
        # 初始化envelope_model为本地模型深拷贝
        self.envelope_model = copy.deepcopy(model)
        self.envelope_model.to(self.device)
        for param in self.envelope_model.parameters():
            param.requires_grad = False

        pgd_attack = PGD_Linf_Attack(model,
                                     epsilon=self.epsilon,
                                     alpha=0.01,  # 固定步长
                                     steps=self.pgd_steps,  # 固定迭代次数
                                     random_start=True)

        swa_n = 0
        for _ in range(self.args.local_ep):
            for images, labels in self.trainloader:
                images, labels = images.to(self.device), labels.to(self.device)
                model.zero_grad()
                outputs = model(images)
                clean_loss = self.criterion(outputs, labels)
                if self.epsilon == 0.0:
                    total_loss = clean_loss
                else:
                    adv_images = pgd_attack.perturb(images, labels)
                    adv_outputs = model(adv_images)
                    adv_loss = self.criterion(adv_outputs, labels)
                    adv_weight = 2 * torch.sigmoid(torch.tensor(5.0 * self.epsilon)).item()
                    clean_weight = 1.0
                    total_loss = clean_weight * clean_loss + adv_weight * adv_loss
                # ---- ME-A Proximal loss ----
                prox_loss = self.rho / 2 * prox_term(model, self.envelope_model)
                total_loss = total_loss + prox_loss

                total_loss.backward()
                optimizer.step()
                # ---- ME-A包络参数更新 ----
                swa_n += 1
                beta = self.me_alpha
                step_u(self.envelope_model, model, beta)

        # 只返回主模型的参数即可（全局聚合还是原来的流程）
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

# ============ 全局测试相关函数保持不变 ============

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
    model.eval()
    device = 'cuda' if args.gpu else 'cpu'
    criterion = nn.NLLLoss().to(device)
    testloader = DataLoader(test_dataset, batch_size=128, shuffle=False)
    results = {}

    for epsilon in epsilon_list:
        pgd_attack = PGD_Linf_Attack(
            model=model,
            epsilon=epsilon,
            alpha=0.01,
            steps=7
        )
        total_loss = 0.0
        total, correct = 0.0, 0.0

        for batch_idx, (images, labels) in enumerate(testloader):
            images, labels = images.to(device), labels.to(device)
            adv_images = pgd_attack.perturb(images, labels)
            with torch.no_grad():
                outputs = model(adv_images)
                loss = criterion(outputs, labels)
            total_loss += loss.item()
            _, pred_labels = torch.max(outputs, 1)
            correct += torch.sum(torch.eq(pred_labels, labels)).item()
            total += len(labels)
        accuracy = correct / total
        avg_loss = total_loss / len(testloader)
        results[f'epsilon_{epsilon}'] = {
            'accuracy': accuracy,
            'loss': avg_loss
        }
        print(f'ε={epsilon:.1f} | Test Acc: {accuracy:.2%} | Loss: {avg_loss:.4f}')
    return results

class LRScheduler:
    def __init__(self, optimizer, args):
        self.optimizer = optimizer
        self.args = args
        self.current_lr = args.lr
        self.global_round = 0
        self.last_decay_round = 0
        self.warmup_epochs = 5
        self.warmup_lr = args.lr * 0.1

    def step(self):
        self.global_round += 1
        if self.global_round <= self.warmup_epochs:
            self.current_lr = self.warmup_lr + (self.args.lr - self.warmup_lr) * \
                              (self.global_round / self.warmup_epochs)
        else:
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
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.current_lr
        return self.current_lr
