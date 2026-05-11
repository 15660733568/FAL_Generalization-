# options.py
#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse

def args_parser():
    parser = argparse.ArgumentParser()

    # Federated args
    parser.add_argument('--epochs', type=int, default=50, help="number of global rounds")
    parser.add_argument('--num_users', type=int, default=20, help="number of users (clients)")
    parser.add_argument('--frac', type=float, default=0.5, help='fraction of clients per round')
    parser.add_argument('--local_ep', type=int, default=10, help="local epochs per client")
    parser.add_argument('--local_bs', type=int, default=64, help="local batch size")
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
    parser.add_argument('--momentum', type=float, default=0.9, help='SGD momentum')
    parser.add_argument('--optimizer', type=str, default='sgd', help="sgd/adam")
    parser.add_argument('--lr_decay', type=float, default=0.9, help='lr decay rate')
    parser.add_argument('--lr_decay_epoch', type=int, default=5, help='decay every N rounds')

    # Adversarial args
    parser.add_argument('--adv_train', type=int, default=0, help='1 to enable adversarial training')
    parser.add_argument('--epsilon', type=float, default=0.0, help='PGD epsilon (e.g., 4/255)')
    parser.add_argument('--alpha', type=float, default=0.0, help='PGD alpha (step)')
    parser.add_argument('--attack_iters', type=int, default=10, help='PGD iterations')
    parser.add_argument('--restarts', type=int, default=1, help='PGD restarts')
    parser.add_argument('--attack_norm', type=str, default='Linf', help='PGD norm (Linf)')

    # Dataset/Model
    parser.add_argument('--dataset', type=str, default='susy', help="susy")
    parser.add_argument('--model', type=str, default='cnn', help='cnn')
    parser.add_argument('--num_classes', type=int, default=2, help="2 for SUSY")
    parser.add_argument('--num_channels', type=int, default=1, help="1 for SUSY (tabular as 1x18)")
    parser.add_argument('--iid', type=int, default=1, help='1 for IID, 0 for non-IID')
    parser.add_argument('--unequal', type=int, default=0, help='unused for IID')

    # SUSY paths & auto-download
    parser.add_argument('--susy_root', type=str, default='../data/SUSY.csv',
                        help='path to SUSY.csv, or a directory to place SUSY.csv, or a compressed file path')
    parser.add_argument('--auto_download', type=int, default=1,
                        help='auto-download SUSY if missing (1=yes,0=no)')
    parser.add_argument('--susy_url', type=str,
                        default='https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/SUSY.csv.bz2',
                        help='primary source URL of SUSY dataset (.csv.gz/.csv.bz2)')
    parser.add_argument('--insecure', type=int, default=0,
                        help='skip TLS verification for dataset download (NOT recommended)')

    # Misc
    parser.add_argument('--gpu', type=int, default=0, help="GPU id, set to -1 for CPU")
    parser.add_argument('--stopping_rounds', type=int, default=10, help='early stopping (unused)')
    parser.add_argument('--verbose', type=int, default=1, help='verbose flag')
    parser.add_argument('--seed', type=int, default=1, help='random seed')

    return parser.parse_args()
