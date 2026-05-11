# update.py
#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from attack import pgd_linf

class DatasetSplit(Dataset):
    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = [int(i) for i in idxs]

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        # 保持你原有“转 tensor”的细节（即使 dataset 已返回 tensor 也无妨）
        return torch.tensor(image), torch.tensor(label)

class LocalUpdate(object):
    def __init__(self, args, dataset, idxs, logger, attack_params=None, device=None, client_id=None):
        self.args = args
        self.logger = logger
        self.attack_params = attack_params
        self.trainloader, self.validloader, self.testloader = self.train_val_test(dataset, list(idxs))
        self.device = device if device is not None else (torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu'))
        self.criterion = nn.CrossEntropyLoss().to(self.device)
        self.client_id = client_id  # 可选

    def train_val_test(self, dataset, idxs):
        idxs_train = idxs[:int(0.8*len(idxs))]
        idxs_val = idxs[int(0.8*len(idxs)):int(0.9*len(idxs))]
        idxs_test = idxs[int(0.9*len(idxs)):]
        trainloader = DataLoader(DatasetSplit(dataset, idxs_train), batch_size=self.args.local_bs, shuffle=True)
        validloader = DataLoader(DatasetSplit(dataset, idxs_val), batch_size=max(1, int(len(idxs_val)/10)), shuffle=False)
        testloader  = DataLoader(DatasetSplit(dataset, idxs_test), batch_size=max(1, int(len(idxs_test)/10)), shuffle=False)
        return trainloader, validloader, testloader

    def update_weights(self, model, global_round, current_lr=None):
        model.to(self.device)
        model.train()
        epoch_loss = []
        lr = current_lr if current_lr is not None else self.args.lr

        if self.args.optimizer == 'sgd':
            optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
        elif self.args.optimizer == 'adam':
            optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
        else:
            raise ValueError(f"Unknown optimizer: {self.args.optimizer}")

        for local_ep in range(self.args.local_ep):
            batch_loss = []
            for _, (images, labels) in enumerate(self.trainloader):
                images, labels = images.to(self.device), labels.to(self.device)

                # 若开启对抗训练，生成对抗样本（使用 PGD-Linf）
                if self.attack_params is not None:
                    eps = self.attack_params['epsilon']
                    alpha = self.attack_params.get('alpha', 1/255)
                    iters = self.attack_params.get('iters', 10)
                    images = pgd_linf(model, images, labels, eps, alpha, iters)

                model.zero_grad()
                logits = model(images)
                loss = self.criterion(logits, labels)
                loss.backward()
                optimizer.step()
                batch_loss.append(loss.item())

            local_epoch_loss = sum(batch_loss) / max(1, len(batch_loss))
            epoch_loss.append(local_epoch_loss)
            print(f"      [Local Epoch {local_ep + 1}/{self.args.local_ep}] Loss: {local_epoch_loss:.4f}")

        return model.state_dict(), sum(epoch_loss) / max(1, len(epoch_loss))

    def inference(self, model):
        model.to(self.device)
        model.eval()
        loss, total, correct = 0.0, 0, 0
        for _, (images, labels) in enumerate(self.testloader):
            images, labels = images.to(self.device), labels.to(self.device)
            outputs = model(images)
            batch_loss = self.criterion(outputs, labels)
            loss += batch_loss.item()
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
        acc = correct / max(1, total)
        return acc, loss

def test_inference(args, model, test_dataset, attack_params=None, device=None):
    device = device or (torch.device('cuda:0') if torch.cuda.is_available() and args.gpu >= 0 else torch.device('cpu'))
    model.to(device)
    model.eval()
    loss, total, correct = 0.0, 0, 0
    criterion = nn.CrossEntropyLoss().to(device)
    testloader = DataLoader(test_dataset, batch_size=128, shuffle=False)

    for _, (images, labels) in enumerate(testloader):
        images, labels = images.to(device), labels.to(device)
        if attack_params is not None:
            eps = attack_params['epsilon']
            alpha = attack_params.get('alpha', 1/255)
            iters = attack_params.get('iters', 10)
            images = pgd_linf(model, images, labels, eps, alpha, iters)
        outputs = model(images)
        batch_loss = criterion(outputs, labels)
        loss += batch_loss.item()
        _, preds = torch.max(outputs, 1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
    acc = correct / max(1, total)
    return acc, loss

def adversarial_test(model, dataset, epsilon, device, iters=10, alpha=None, norm='Linf'):
    model.eval()
    model.to(device)
    testloader = DataLoader(dataset, batch_size=128, shuffle=False)
    if alpha is None:
        alpha = epsilon / 4.0

    correct, total = 0, 0
    for _, (images, labels) in enumerate(testloader):
        images, labels = images.to(device), labels.to(device)
        adv_images = pgd_linf(model, images, labels, epsilon, alpha, iters)
        outputs = model(adv_images)
        _, preds = torch.max(outputs, 1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
    acc = correct / max(1, total)
    print(f"[Adversarial Test] ε={epsilon:.6f}, iters={iters}, acc={acc:.4f}")
    return acc
