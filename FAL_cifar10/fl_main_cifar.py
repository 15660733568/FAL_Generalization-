#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

import os
import copy
import time
import pickle
import numpy as np
from tqdm import tqdm

import torch
from tensorboardX import SummaryWriter

from options import args_parser
from update import LocalUpdate, test_inference
from models import MLP, CNNMnist, CNNFashion_Mnist,ImprovedCNN
from utils import get_dataset, average_weights, exp_details
from model_cifar import CNNCifar

if __name__ == '__main__':
    start_time = time.time()

    # define paths
    path_project = os.path.abspath('..')
    logger = SummaryWriter('../logs')

    args = args_parser()
    exp_details(args)

    if args.gpu:
        torch.cuda.set_device(args.gpu)
    device = 'cuda' if args.gpu else 'cpu'

    # load dataset and user groups
    train_dataset, test_dataset, user_groups = get_dataset(args)#加载训练集、测试集和用户组

    # BUILD MODEL
    if args.model == 'cnn':
        if args.dataset == 'mnist':
            global_model = CNNMnist(args=args)
        elif args.dataset == 'fmnist':
            global_model = CNNFashion_Mnist(args=args)
        elif args.dataset == 'cifar':
            global_model = CNNCifar()
        elif args.dataset == 'svhn':
            global_model = ImprovedCNN(num_classes=10)  # SVHN有10个类别

    elif args.model == 'mlp':
        # Multi-layer preceptron
        img_size = train_dataset[0][0].shape
        len_in = 1
        for x in img_size:
            len_in *= x
            global_model = MLP(dim_in=len_in, dim_hidden=64,
                               dim_out=args.num_classes)
    else:
        exit('Error: unrecognized model')

    # Set the model to train and send it to device.
    global_model.to(device)
    global_model.train()
    print(global_model)#将全局模型发送到指定的设备上，并将其设置为训练模式。打印全局模型的结构。

    # copy weights
    global_weights = global_model.state_dict()#复制全局模型的结构，初始化全局模型global_model，并获取其初始权重global_weights，相当于算法 1 中初始化w0

    # Training
    train_loss, train_accuracy = [], []
    val_acc_list, net_list = [], []
    cv_loss, cv_acc = [], []
    print_every = 2
    val_loss_pre, counter = 0, 0

    # 在训练循环前添加当前学习率变量
    current_lr = args.lr

    for epoch in tqdm(range(args.epochs)):
        # 学习率衰减逻辑
        if (epoch + 1) % args.lr_decay_epoch == 0:
            current_lr *= args.lr_decay
            print(f'\nLearning rate decayed to: {current_lr}\n')

        local_weights, local_losses = [], []
        print(f'\n | Global Training Round : {epoch + 1} |\n')

        global_model.train()
        m = max(int(args.frac * args.num_users), 1)
        idxs_users = np.random.choice(range(args.num_users), m, replace=False)

        for idx in idxs_users:
            local_model = LocalUpdate(args=args, dataset=train_dataset,
                                      idxs=user_groups[idx], logger=logger)
            # 传递当前学习率
            w, loss = local_model.update_weights(
                model=copy.deepcopy(global_model),
                global_round=epoch,
                current_lr=current_lr)

            local_weights.append(copy.deepcopy(w))
            local_losses.append(copy.deepcopy(loss))
        # update global weights
        #通过average_weights函数对本地更新后的权重进行平均，得到新的全局权重global_weights，
        global_weights = average_weights(local_weights)
        # 并将其加载到全局模型中，相当于算法 1 中服务器根据客户端的更新结果进行模型平均和更新。
        # update global weights
        global_model.load_state_dict(global_weights)

        loss_avg = sum(local_losses) / len(local_losses)
        train_loss.append(loss_avg)
        # Calculate avg training accuracy over all users at every epoch
        list_acc, list_loss = [], []
        global_model.eval()
        for c in range(args.num_users):
            local_model = LocalUpdate(args=args, dataset=train_dataset,
                                      idxs=user_groups[idx], logger=logger)
            acc, loss = local_model.inference(model=global_model)
            list_acc.append(acc)
            list_loss.append(loss)
        train_accuracy.append(sum(list_acc)/len(list_acc))

        # print global training loss after every 'i' rounds
        if ( epoch ) % print_every == 0:
            print(f' \nAvg Training Stats after {epoch+1} global rounds:')
            print(f'Training Loss : {np.mean(np.array(train_loss))}')
            print('Train Accuracy: {:.2f}% \n'.format(100*train_accuracy[-1]))

    # Test inference after completion of training
    test_acc, test_loss = test_inference(args, global_model, test_dataset)

    print(f' \n Results after {args.epochs} global rounds of training:')
    print("|---- Avg Train Accuracy: {:.2f}%".format(100*train_accuracy[-1]))
    print("|---- Test Accuracy: {:.2f}%".format(100*test_acc))

    # Saving the objects train_loss and train_accuracy:
    file_name = '../fed/objects/{}_{}_{}_C[{}]_iid[{}]_E[{}]_B[{}].pkl'.\
        format(args.dataset, args.model, args.epochs, args.frac, args.iid,
               args.local_ep, args.local_bs)

    with open(file_name, 'wb') as f:
        pickle.dump([train_loss, train_accuracy], f)

    print('\n Total Run Time: {0:0.4f}'.format(time.time()-start_time))

    #PLOTTING (optional)
    import matplotlib
    import matplotlib.pyplot as plt
    matplotlib.use('Agg')

    #Plot Loss curve
    plt.figure()
    plt.title('Training Loss vs Communication rounds')
    plt.plot(range(len(train_loss)), train_loss, color='r')
    plt.ylabel('Training loss')
    plt.xlabel('Communication Rounds')
    plt.savefig('../fed/fed_{}_{}_{}_C[{}]_iid[{}]_E[{}]_B[{}]_loss.png'.
               format(args.dataset, args.model, args.epochs, args.frac,
                    args.iid, args.local_ep, args.local_bs))

    #  Plot Average Accuracy vs Communication rounds
    plt.figure()
    plt.title('Average Accuracy vs Communication rounds')
    plt.plot(range(len(train_accuracy)), train_accuracy, color='k')
    plt.ylabel('Average Accuracy')
    plt.xlabel('Communication Rounds')
    plt.savefig('../fed/fed_{}_{}_{}_C[{}]_iid[{}]_E[{}]_B[{}]_acc.png'.
               format(args.dataset, args.model, args.epochs, args.frac,
                      args.iid, args.local_ep, args.local_bs))
