#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import copy
import time
import pickle
import numpy as np
from tqdm import tqdm
import torch
from tensorboardX import SummaryWriter
import matplotlib.pyplot as plt

from options import args_parser
from update_ME import LocalUpdate, test_inference, test_adv_inference, LRScheduler
from models import MLP, CNNMnist, CNNFashion_Mnist, CNNCifar
from utils import get_dataset, average_weights, exp_details


def plot_adv_results(adv_results_all_runs):
    plt.figure(figsize=(15, 6))
    plt.rcParams['font.size'] = 12
    plt.rcParams['axes.labelsize'] = 12
    plt.rcParams['xtick.labelsize'] = 11
    plt.rcParams['ytick.labelsize'] = 11

    epsilons = sorted(adv_results_all_runs[0].keys())
    num_runs = len(adv_results_all_runs)
    num_epochs = len(adv_results_all_runs[0][epsilons[0]]['test_acc'])

    mean_test_acc = {epsilon: np.zeros(num_epochs) for epsilon in epsilons}
    std_test_acc = {epsilon: np.zeros(num_epochs) for epsilon in epsilons}
    mean_gap = {epsilon: np.zeros(num_epochs) for epsilon in epsilons}
    std_gap = {epsilon: np.zeros(num_epochs) for epsilon in epsilons}

    for epsilon in epsilons:
        test_acc_runs = np.array([adv_results_all_runs[run][epsilon]['test_acc'] for run in range(num_runs)])
        gap_runs = np.array([adv_results_all_runs[run][epsilon]['gap'] for run in range(num_runs)])
        mean_test_acc[epsilon] = np.mean(test_acc_runs, axis=0)
        std_test_acc[epsilon] = np.std(test_acc_runs, axis=0)
        mean_gap[epsilon] = np.mean(gap_runs, axis=0)
        std_gap[epsilon] = np.std(gap_runs, axis=0)

    colors = ['#1f77b4', '#2ca02c', '#d62728', '#9467bd', '#ff7f0e', '#8c564b']
    linestyles = ['-', '--', '-.', ':', '-', '--']
    markers = ['o', 's', '^', 'D', 'v', '*']
    markersizes = [6, 5, 6, 5, 6, 6]

    # --------- (1) 测试精度曲线加误差棒棒 ---------
    plt.subplot(1, 2, 1)
    for i, epsilon in enumerate(epsilons):
        epochs = np.arange(1, num_epochs + 1)
        # 先画完整的曲线
        plt.plot(epochs, mean_test_acc[epsilon],
                 color=colors[i], linestyle=linestyles[i],
                 label=f'ε={epsilon}', linewidth=2)
        # 再画每个点的误差棒
        plt.errorbar(
            epochs, mean_test_acc[epsilon],
            yerr=std_test_acc[epsilon],
            fmt=markers[i],  # 点的形状
            color=colors[i],
            markersize=markersizes[i],
            capsize=3, elinewidth=1.5, linestyle='none',
            alpha=0.85
        )
    plt.title('Test Accuracy vs Epochs', pad=15)
    plt.xlabel('Epochs', labelpad=8)
    plt.ylabel('Accuracy', labelpad=8)
    plt.ylim(0.70, 0.95)
    plt.xticks(np.arange(0, num_epochs + 1, 5))
    plt.yticks(np.arange(0.70, 0.96, 0.05))
    plt.legend(loc='lower right', framealpha=1)
    plt.grid(True, linestyle='--', alpha=0.6)

    # --------- (2) 泛化gap曲线加误差棒棒 ---------
    plt.subplot(1, 2, 2)
    for i, epsilon in enumerate(epsilons):
        epochs = np.arange(1, num_epochs + 1)
        plt.plot(
            epochs, mean_gap[epsilon],
            color=colors[i], linestyle=linestyles[i],
            label=f'ε={epsilon}', linewidth=2
        )
        plt.errorbar(
            epochs, mean_gap[epsilon],
            yerr=std_gap[epsilon],
            fmt=markers[i],
            color=colors[i],
            markersize=markersizes[i],
            capsize=3, elinewidth=1.5, linestyle='none',
            alpha=0.85
        )
    plt.title('Generalization Gap vs Epochs', pad=15)
    plt.xlabel('Epochs', labelpad=8)
    plt.ylabel('Gap', labelpad=8)
    plt.ylim(0.00, 0.14)
    plt.xticks(np.arange(0, num_epochs + 1, 5))
    plt.yticks(np.arange(0.00, 0.15, 0.02))
    plt.legend(loc='upper right', framealpha=1)
    plt.grid(True, linestyle='--', alpha=0.6)

    plt.tight_layout()
    os.makedirs('../save', exist_ok=True)
    plt.savefig('../save/adv_training_comparison_errorbars.png', dpi=300, bbox_inches='tight')
    plt.close()

def run_experiment(args, train_dataset, user_groups, test_dataset, device, logger=None):
    """Run a single experiment and return results and models"""
    adv_results = {epsilon: {'train_acc': [], 'test_acc': [], 'gap': []}
                   for epsilon in args.epsilon_list}

    # Model initialization for each epsilon
    global_models = {}
    for epsilon in args.epsilon_list:
        if args.model == 'cnn':
            if args.dataset == 'mnist':
                global_models[epsilon] = CNNMnist(args=args)
            elif args.dataset == 'fmnist':
                global_models[epsilon] = CNNFashion_Mnist(args=args)
            elif args.dataset == 'cifar':
                global_models[epsilon] = CNNCifar(args=args)
        elif args.model == 'mlp':
            img_size = train_dataset[0][0].shape
            len_in = 1
            for x in img_size:
                len_in *= x
            global_models[epsilon] = MLP(dim_in=len_in, dim_hidden=64,
                                         dim_out=args.num_classes)
        global_models[epsilon].to(device)
        global_models[epsilon].train()

    # Training loop
    for epoch in tqdm(range(args.epochs)):
        for epsilon in args.epsilon_list:
            current_epsilon = min(epsilon * (epoch + 1) / args.epochs, epsilon)

            # Local training
            local_weights, local_losses = [], []
            m = max(int(args.frac * args.num_users), 1)
            idxs_users = np.random.choice(range(args.num_users), m, replace=False)

            for idx in idxs_users:
                local_model = LocalUpdate(
                    args=args,
                    dataset=train_dataset,
                    idxs=user_groups[idx],
                    logger=logger,
                    epsilon=current_epsilon
                )
                w, loss = local_model.update_weights(
                    model=copy.deepcopy(global_models[epsilon]),
                    global_round=epoch
                )
                local_weights.append(copy.deepcopy(w))
                local_losses.append(copy.deepcopy(loss))

            # Model aggregation
            global_weights = average_weights(local_weights)
            global_models[epsilon].load_state_dict(global_weights)

            # Evaluation
            train_acc, _ = test_inference(args, global_models[epsilon], train_dataset)
            test_acc = test_adv_inference(args, global_models[epsilon], test_dataset, [epsilon])

            # Record results
            adv_results[epsilon]['train_acc'].append(train_acc)
            adv_results[epsilon]['test_acc'].append(test_acc[f'epsilon_{epsilon}']['accuracy'])
            adv_results[epsilon]['gap'].append(
                abs(train_acc - test_acc[f'epsilon_{epsilon}']['accuracy'])
            )

            # Logging
            if (epoch + 1) % args.print_every == 0:
                print(f'ε={epsilon} | Epoch {epoch + 1}/{args.epochs} | '
                      f'Train: {train_acc:.2%} | Test: {test_acc[f"epsilon_{epsilon}"]["accuracy"]:.2%} | '
                      f'Gap: {adv_results[epsilon]["gap"][-1]:.4f}')

    return adv_results, global_models

# if __name__ == '__main__':
#     start_time = time.time()
#     os.makedirs('save', exist_ok=True)
#     logger = SummaryWriter('logs')  # 这里建议logs放到本地目录，避免路径混乱
#     args = args_parser()
#     args.epsilon_list = [0.0, 0.1, 0.3, 0.5]  # 可自定义
#     exp_details(args)
#     device = 'cuda' if args.gpu else 'cpu'
#
#     # Load data
#     train_dataset, test_dataset, user_groups = get_dataset(args)
#
#     # Run experiment multiple times
#     num_runs = 5
#     all_results = []
#
#     for run in range(num_runs):
#         print(f'\n=== Starting Run {run + 1}/{num_runs} ===')
#         run_results, run_models = run_experiment(
#             args, train_dataset, user_groups, test_dataset, device, logger
#         )
#         all_results.append(run_results)
#
#         # 保存每次实验所有中间信息（推荐做法，防止实验中断信息丢失）
#         with open(f'save/adv_results_run{run + 1}.pkl', 'wb') as f:
#             pickle.dump(run_results, f)
#
#         # 保存每个模型（可选）
#         os.makedirs('save/models', exist_ok=True)
#         for epsilon in args.epsilon_list:
#             torch.save(run_models[epsilon].state_dict(),
#                        f'save/models/{args.dataset}_eps{epsilon}_run{run + 1}_final.pt')
#
#     # 保存全部实验信息
#     with open('save/adv_results_all_runs.pkl', 'wb') as f:
#         pickle.dump(all_results, f)
#
#     print(f'\nTotal Run Time: {time.time() - start_time:.2f}s')
def main_falme(num_runs=5):
    """
    运行 FALME 实验 num_runs 次，返回总运行时间（秒）

    参数：
        num_runs (int): 重复完整训练+测试的次数
    返回：
        total_time (float): 总耗时（秒）
    """
    start_time = time.time()

    # 日志和保存目录
    os.makedirs('save', exist_ok=True)
    logger = SummaryWriter('logs')  # 日志目录

    # 参数
    args = args_parser()
    # 为了和普通 FAL 对齐，建议这里的 epsilon_list 和普通 FAL 保持完全一致
    args.epsilon_list = [0.0, 0.1, 0.3, 0.5]  # 你可以按需要改，但两边要一致
    exp_details(args)

    # 设备
    device = 'cuda' if (args.gpu and torch.cuda.is_available()) else 'cpu'

    # 数据
    train_dataset, test_dataset, user_groups = get_dataset(args)

    # 多次完整实验
    all_results = []
    for run in range(num_runs):
        print(f'\n=== FALME: Starting Run {run + 1}/{num_runs} ===')
        run_results, run_models = run_experiment(
            args, train_dataset, user_groups, test_dataset, device, logger
        )
        all_results.append(run_results)

        # 保存每次实验的结果
        with open(f'save/adv_results_run{run + 1}.pkl', 'wb') as f:
            pickle.dump(run_results, f)

        # （可选）保存每次实验的模型
        os.makedirs('save/models', exist_ok=True)
        for epsilon in args.epsilon_list:
            torch.save(
                run_models[epsilon].state_dict(),
                f'save/models/{args.dataset}_eps{epsilon}_run{run + 1}_final.pt'
            )

    # 保存全部实验信息
    with open('save/adv_results_all_runs.pkl', 'wb') as f:
        pickle.dump(all_results, f)

    total_time = time.time() - start_time
    print(f'\n[FALME] Total Run Time: {total_time:.2f}s')
    return total_time


if __name__ == '__main__':
    # 直接运行这个文件时，默认跑 3 次完整实验
    main_falme(num_runs=5)