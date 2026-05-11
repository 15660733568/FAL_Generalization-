#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import argparse


def args_parser():
    parser = argparse.ArgumentParser()

    # federated arguments
    parser.add_argument('--epochs', type=int, default=20,
                        help="number of rounds of training")
    parser.add_argument('--num_users', type=int, default=100,
                        help="number of users: K")
    parser.add_argument('--frac', type=float, default=0.5,
                        help='the fraction of clients: C')
    parser.add_argument('--local_ep', type=int, default=5,
                        help="the number of local epochs: E")
    parser.add_argument('--local_bs', type=int, default=10,
                        help="local batch size: B")
    parser.add_argument('--lr', type=float, default=0.01,
                        help='learning rate')
    parser.add_argument('--momentum', type=float, default=0.5,
                        help='SGD momentum (default: 0.5)')
    parser.add_argument('--pgd_steps', type=int, default=10,
                        help='number of PGD steps for adversarial training')

    # model arguments
    parser.add_argument('--model', type=str, default='cnn', help='model name')
    parser.add_argument('--kernel_num', type=int, default=9,
                        help='number of each kind of kernel')
    parser.add_argument('--kernel_sizes', type=str, default='3,4,5',
                        help='comma-separated kernel size to use for convolution')
    parser.add_argument('--num_channels', type=int, default=1,
                        help="number of channels of imgs")
    parser.add_argument('--norm', type=str, default='batch_norm',
                        help="batch_norm, layer_norm, or None")
    parser.add_argument('--num_filters', type=int, default=32,
                        help="number of filters for conv nets")

    # learning rate decay
    parser.add_argument('--lr_decay', type=str, default='step',
                        help='learning rate decay type: step, exp, cosine')
    parser.add_argument('--lr_decay_step', type=int, default=10,
                        help='number of epochs after which to decay lr')
    parser.add_argument('--lr_decay_rate', type=float, default=0.1,
                        help='learning rate decay rate')
    parser.add_argument('--min_lr', type=float, default=1e-5,
                        help='minimum learning rate')

    # other arguments
    parser.add_argument('--dataset', type=str, default='mnist',
                        help="name of dataset")
    parser.add_argument('--num_classes', type=int, default=10,
                        help="number of classes")
    parser.add_argument('--gpu', default=None,
                        help="To use cuda, set to a specific GPU ID")
    parser.add_argument('--optimizer', type=str, default='sgd',
                        help="type of optimizer")
    parser.add_argument('--iid', type=int, default=1,
                        help='Default set to IID. Set to 0 for non-IID.')
    parser.add_argument('--unequal', type=int, default=0,
                        help='whether to use unequal data splits')
    parser.add_argument('--stopping_rounds', type=int, default=10,
                        help='rounds of early stopping')
    parser.add_argument('--verbose', type=int, default=1,
                        help='verbose')
    parser.add_argument('--seed', type=int, default=1,
                        help='random seed')
    parser.add_argument('--print_every', type=int, default=1,
                        help='print results every N epochs')

    args = parser.parse_args()
    return args