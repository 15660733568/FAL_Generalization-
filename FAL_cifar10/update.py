#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from attacks import PGDAttack

class DatasetSplit(Dataset):
    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = [int(i) for i in idxs]

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        if isinstance(self.dataset, torch.utils.data.TensorDataset):
            image, label = self.dataset[self.idxs[item]]
            return image, label
        else:
            image, label = self.dataset[self.idxs[item]]
            return torch.tensor(image), torch.tensor(label)

class LocalUpdate(object):
    def __init__(self, args, dataset, idxs, logger):
        self.args = args
        self.logger = logger
        self.trainloader, self.validloader, self.testloader = self.train_val_test(dataset, list(idxs))
        self.device = 'cuda' if args.gpu else 'cpu'
        self.criterion = nn.CrossEntropyLoss().to(self.device)

    def generate_adv_examples(self, model, images, labels):
        attack = PGDAttack(model, epsilon=self.args.epsilon,
                      alpha=self.args.alpha,
                      iters=self.args.attack_iters,
                      restarts=self.args.restarts,
                      norm=self.args.attack_norm,
                      num_classes=self.args.num_classes)
        return attack.perturb(images, labels)

    def train_val_test(self, dataset, idxs):
        idxs_train = idxs[:int(0.8 * len(idxs))]
        idxs_val = idxs[int(0.8 * len(idxs)):int(0.9 * len(idxs))]
        idxs_test = idxs[int(0.9 * len(idxs)):]

        trainloader = DataLoader(DatasetSplit(dataset, idxs_train), batch_size=self.args.local_bs, shuffle=True)
        validloader = DataLoader(DatasetSplit(dataset, idxs_val), batch_size=int(len(idxs_val)/10), shuffle=False)
        testloader = DataLoader(DatasetSplit(dataset, idxs_test), batch_size=int(len(idxs_test)/10), shuffle=False)
        return trainloader, validloader, testloader

    def update_weights(self, model, global_round, current_lr=None):
        model.train()
        epoch_loss = []
        lr = current_lr if current_lr is not None else self.args.lr

        if self.args.optimizer == 'sgd':
            optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=self.args.momentum)
        elif self.args.optimizer == 'adam':
            optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)

        for _ in range(self.args.local_ep):
            batch_loss = []
            for images, labels in self.trainloader:
                images, labels = images.to(self.device), labels.to(self.device)

                # ✅ 纯对抗训练：仅使用对抗样本
                if self.args.adv_train:
                    model.eval()
                    adv_images = self.generate_adv_examples(model, images, labels)
                    model.train()
                    train_images, train_labels = adv_images, labels
                else:
                    train_images, train_labels = images, labels

                model.zero_grad()
                outputs = model(train_images)
                loss = self.criterion(outputs, train_labels)
                loss.backward()
                optimizer.step()

                self.logger.add_scalar('loss', loss.item())
                batch_loss.append(loss.item())

            epoch_loss.append(sum(batch_loss) / len(batch_loss))

        return model.state_dict(), sum(epoch_loss) / len(epoch_loss)

    def inference(self, model):
        model.eval()
        loss, total, correct = 0.0, 0.0, 0.0

        for images, labels in self.testloader:
            images, labels = images.to(self.device), labels.to(self.device)
            outputs = model(images)
            batch_loss = self.criterion(outputs, labels)
            loss += batch_loss.item()

            _, pred_labels = torch.max(outputs, 1)
            correct += torch.sum(pred_labels == labels).item()
            total += len(labels)

        accuracy = correct / total
        return accuracy, loss

def test_inference(args, model, test_dataset):
    model.eval()
    loss, total, correct = 0.0, 0.0, 0.0
    device = 'cuda' if args.gpu else 'cpu'
    criterion = nn.CrossEntropyLoss().to(device)
    testloader = DataLoader(test_dataset, batch_size=128, shuffle=False)

    for images, labels in testloader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        batch_loss = criterion(outputs, labels)
        loss += batch_loss.item()
        _, preds = torch.max(outputs, 1)
        correct += torch.sum(preds == labels).item()
        total += len(labels)

    accuracy = correct / total
    return accuracy, loss

def adversarial_test(model, test_loader_or_dataset, epsilon, device='cpu', iters=10, alpha=None, norm='Linf'):
    import torch
    from torch.utils.data import DataLoader
    from torch import nn

    model.eval()
    device = device or ('cuda' if next(model.parameters()).is_cuda else 'cpu')
    criterion = nn.CrossEntropyLoss().to(device)

    if isinstance(test_loader_or_dataset, DataLoader):
        testloader = test_loader_or_dataset
    else:
        testloader = DataLoader(test_loader_or_dataset, batch_size=128, shuffle=False)

    step = alpha if alpha is not None else float(epsilon) / 4.0
    attacker = PGDAttack(model, epsilon=float(epsilon), alpha=float(step), iters=int(iters), restarts=1, norm=norm, num_classes=getattr(model, 'num_classes', 10))

    total, correct = 0, 0
    for images, labels in testloader:
        images, labels = images.to(device), labels.to(device)
        adv_images = attacker.perturb(images, labels)
        with torch.no_grad():
            outputs = model(adv_images)
            _, preds = outputs.max(1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    return correct / total
