#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np

def susy_iid(num_samples, num_users, seed=1):
    """将样本均匀分给各客户端（简单 IID 划分）。"""
    rng = np.random.RandomState(seed)
    all_idx = rng.permutation(num_samples)
    per = num_samples // num_users
    groups = {}
    for i in range(num_users):
        start = i * per
        end = (i + 1) * per if i < num_users - 1 else num_samples
        groups[i] = set(all_idx[start:end].tolist())
    return groups

def tiny_noniid(dataset, num_users, shards_per_user=2, num_shards=1000):
    """
    A simple shard-based non-IID splitter for Tiny-ImageNet.
    Assumes dataset length divisible by num_shards.
    """
    num_imgs = len(dataset) // num_shards
    idx_shard = [i for i in range(num_shards)]
    dict_users = {i: np.array([]) for i in range(num_users)}
    idxs = np.arange(num_shards * num_imgs)
    labels = np.array(getattr(dataset, 'targets', [dataset[i][1] for i in range(len(dataset))]))

    # sort by label for shard grouping
    idxs_labels = np.vstack((idxs, labels))
    idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
    idxs = idxs_labels[0, :]

    for i in range(num_users):
        rand_set = set(np.random.choice(idx_shard, shards_per_user, replace=False))
        idx_shard = list(set(idx_shard) - rand_set)
        for rand in rand_set:
            dict_users[i] = np.concatenate((dict_users[i], idxs[rand*num_imgs:(rand+1)*num_imgs]), axis=0)
    return dict_users
