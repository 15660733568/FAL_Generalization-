#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import argparse


def args_parser():
    parser = argparse.ArgumentParser()

    # federated arguments (Notation for the arguments followed from paper)
    parser.add_argument('--lr_decay', type=float, default=0.80,
                        help='learning rate decay rate')
    parser.add_argument('--lr_decay_epoch', type=int, default=1,
                        help='decay learning rate every N epochs')
    parser.add_argument('--epochs', type=int, default=20,
                        help="number of rounds of training")
    parser.add_argument('--num_users', type=int, default=10,
                        help="number of users: K")
    parser.add_argument('--frac', type=float, default=0.2,
                        help='the fraction of clients: C')
    parser.add_argument('--local_ep', type=int, default=10,
                        help="the number of local epochs: E")
    parser.add_argument('--local_bs', type=int, default=20,
                        help="local batch size: B")
    parser.add_argument('--lr', type=float, default=0.01,
                        help='learning rate')
    parser.add_argument('--momentum', type=float, default=0.5,
                        help='SGD momentum (default: 0.5)')

    # adversarial training arguments
    parser.add_argument('--adv_train', type=int, default=1,
                    help='whether to use adversarial training')
    parser.add_argument('--attack_type', type=str, default='pgd_linf',
                    help='type of adversarial attack (pgd_linf)')
    parser.add_argument('--epsilon', type=float, default=0.1,
                    help='perturbation size for adversarial attacks')
    parser.add_argument('--alpha', type=float, default=0.01,
                    help='step size for PGD attack')
    parser.add_argument('--attack_iters', type=int, default=10,
                    help='number of iterations for PGD attack')
    parser.add_argument('--restarts', type=int, default=1,
                    help='number of restarts for PGD attack')
    parser.add_argument('--attack_norm', type=str, default='Linf',  # 修改为attack_norm
                    help='norm type for adversarial attack (Linf, L2, etc.)')
    parser.add_argument('--num_adv_examples', type=int, default=5,
                    help='number of adversarial examples to generate per clean sample')

    # model arguments
    parser.add_argument('--model', type=str, default='cnn',
                        help='model name (cnn, mlp, cnn1d)')
    parser.add_argument('--num_classes', type=int, default=2,
                        help="number of classes")
    parser.add_argument('--kernel_num', type=int, default=9,
                        help='number of each kind of kernel')
    parser.add_argument('--kernel_sizes', type=str, default='3,4,5',
                        help='comma-separated kernel size to use for convolution')
    parser.add_argument('--num_channels', type=int, default=1,
                        help="number of channels of imgs")
    parser.add_argument('--model_norm', type=str, default='batch_norm',  # 修改为model_norm
                        help="model normalization type (batch_norm, layer_norm, or None)")
    parser.add_argument('--num_filters', type=int, default=32,
                        help="number of filters for conv nets")

    # other arguments
    parser.add_argument('--dataset', type=str, default='cifar',
                        help="name of dataset")
    parser.add_argument('--data_path', type=str,
                        default='/home/hyt/FL/data/webspam_wc_normalized_unigram.svm',
                        help="path to dataset file")
    parser.add_argument('--gpu', default=0,
                        help="To use cuda, set to a specific GPU ID. Default set to use CPU.")
    parser.add_argument('--optimizer', type=str, default='sgd',
                        help="type of optimizer")
    parser.add_argument('--iid', type=int, default=1,
                        help='Default set to IID. Set to 0 for non-IID.')
    parser.add_argument('--unequal', type=int, default=0,
                        help='whether to use unequal data splits for non-i.i.d setting')
    parser.add_argument('--stopping_rounds', type=int, default=10,
                        help='rounds of early stopping')
    parser.add_argument('--verbose', type=int, default=1, help='verbose')
    parser.add_argument('--seed', type=int, default=42, help='random seed')
    args = parser.parse_args()
    return args