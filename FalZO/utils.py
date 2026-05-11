#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import copy
import torch
import numpy as np
from torchvision import datasets, transforms
from sampling import mnist_iid, mnist_noniid, mnist_noniid_unequal
from sampling import cifar_iid, cifar_noniid

def get_dataset(args):
    if args.dataset == 'cifar':
        data_dir = '../data/cifar/'
        apply_transform = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        train_dataset = datasets.CIFAR10(data_dir, train=True, download=True, transform=apply_transform)
        test_dataset = datasets.CIFAR10(data_dir, train=False, download=True, transform=apply_transform)
        if args.iid:
            user_groups = cifar_iid(train_dataset, args.num_users)
        else:
            if args.unequal:
                raise NotImplementedError()
            else:
                user_groups = cifar_noniid(train_dataset, args.num_users)

    elif args.dataset == 'mnist' or args.dataset == 'fmnist':
        if args.dataset == 'mnist':
            data_dir = '../data/mnist/'
        else:
            data_dir = '../data/fmnist/'
        apply_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))])
        train_dataset = datasets.MNIST(data_dir, train=True, download=True, transform=apply_transform)
        test_dataset = datasets.MNIST(data_dir, train=False, download=True, transform=apply_transform)
        if args.iid:
            user_groups = mnist_iid(train_dataset, args.num_users)
        else:
            if args.unequal:
                user_groups = mnist_noniid_unequal(train_dataset, args.num_users)
            else:
                user_groups = mnist_noniid(train_dataset, args.num_users)

    return train_dataset, test_dataset, user_groups

def average_weights(w):
    w_avg = copy.deepcopy(w[0])
    for key in w_avg.keys():
        for i in range(1, len(w)):
            w_avg[key] += w[i][key]
        w_avg[key] = torch.div(w_avg[key], len(w))
    return w_avg

def exp_details(args):
    print('\nExperimental details:')
    print(f'    Model     : {args.model}')
    print(f'    Optimizer : {args.optimizer}')
    print(f'    Learning  : {args.lr}')
    print(f'    Global Rounds   : {args.epochs}\n')
    print('    Federated parameters:')
    if args.iid:
        print('    IID')
    else:
        print('    Non-IID')
    print(f'    Fraction of users  : {args.frac}')
    print(f'    Local Batch size   : {args.local_bs}')
    print(f'    Local Epochs       : {args.local_ep}\n')
    return

def softmax(x):
    x = np.array(x)
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

def eval_on_validation(model_class, weights, val_loader, device, args):
    # 自动根据类型初始化模型
    if model_class.__name__ == 'MLP':
        img_size = val_loader.dataset[0][0].shape
        len_in = 1
        for x in img_size:
            len_in *= x
        model = model_class(dim_in=len_in, dim_hidden=64, dim_out=args.num_classes)
    else:
        model = model_class(args=args)
    model.load_state_dict(weights)
    model.to(device)
    model.eval()
    criterion = torch.nn.NLLLoss().to(device)
    loss_sum, count = 0, 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss_sum += loss.item() * images.size(0)
            count += images.size(0)
    return loss_sum / count

def average_weights_with_alpha(local_weights, alpha):
    w_avg = {}
    for key in local_weights[0].keys():
        w_avg[key] = sum(alpha[i] * local_weights[i][key].cpu().float() for i in range(len(local_weights)))
    return w_avg

def zo_aggregate_weights(local_weights, val_loader, model_class, device,
                         steps=15, mu=0.05, num_dir=6, zo_lr=0.1, args=None):
    K = len(local_weights)
    alpha = np.ones(K) / K
    for step in range(steps):
        grad_est = np.zeros(K)
        for _ in range(num_dir):
            u = np.random.randn(K)
            u = u / (np.linalg.norm(u) + 1e-12)
            alpha_pos = softmax(alpha + mu * u)
            alpha_neg = softmax(alpha - mu * u)
            w_pos = average_weights_with_alpha(local_weights, alpha_pos)
            w_neg = average_weights_with_alpha(local_weights, alpha_neg)
            loss_pos = eval_on_validation(model_class, w_pos, val_loader, device, args)
            loss_neg = eval_on_validation(model_class, w_neg, val_loader, device, args)
            grad_est += (loss_pos - loss_neg) / (2 * mu) * u
        grad_est = grad_est / num_dir
        alpha = alpha - zo_lr * grad_est
        alpha = softmax(alpha)
    return average_weights_with_alpha(local_weights, alpha)
