import pickle
import numpy as np
import matplotlib.pyplot as plt
import os

# ---------- 1. 全局大字体和曲线样式 ----------
plt.rcParams['font.size'] = 34
plt.rcParams['axes.titlesize'] = 38
plt.rcParams['axes.labelsize'] = 38
plt.rcParams['xtick.labelsize'] = 34
plt.rcParams['ytick.labelsize'] = 34
plt.rcParams['legend.fontsize'] = 36
plt.rcParams['figure.titlesize'] = 42
plt.rcParams['mathtext.fontset'] = 'stix'  # Latex一致

with open('save/adv_results_all_localep.pkl', 'rb') as f:
    all_results = pickle.load(f)

local_ep_list = sorted(list(all_results.keys()))
delta_list    = [0.0, 0.1, 0.3, 0.5]
num_runs      = len(next(iter(all_results.values())))
num_epochs    = len(next(iter(next(iter(all_results.values()))))[delta_list[0]]['test_acc'])  # 20

# 只保留偶数 epoch：2,4,…,20
even_epochs  = list(range(2, num_epochs + 1, 2))
even_indices = np.arange(1, num_epochs, 2)   # 1,3,…,19

save_dir = 'save'
os.makedirs(save_dir, exist_ok=True)

def pretty_label(delta):
    return rf'$\delta={delta}$'

# ---------- 3. 每个 local_ep：不同 δ 曲线 ----------
for local_ep in local_ep_list:
    # (a) Test Accuracy
    plt.figure(figsize=(16, 12))
    for delta in delta_list:
        test_acc_runs = np.array([all_results[local_ep][run][delta]['test_acc'] for run in range(num_runs)])
        mean_acc = test_acc_runs.mean(axis=0)[even_indices]
        std_acc  = test_acc_runs.std(axis=0)[even_indices]
        plt.errorbar(
            even_epochs, mean_acc, yerr=std_acc, capsize=8, elinewidth=4,
            marker='o', markersize=12, label=pretty_label(delta), linewidth=4, alpha=0.85
        )
    plt.title(f'Test Accuracy (local_ep={local_ep})', fontsize=36)
    plt.xlabel('Epoch', fontsize=40)
    plt.ylabel('Test Accuracy', fontsize=40)
    plt.xticks(even_epochs, fontsize=36)
    plt.yticks([0.6, 0.7, 0.8, 0.9, 1.0], fontsize=36)
    plt.legend(loc='lower right', fontsize=36, frameon=True)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(f'{save_dir}/test_acc_localep{local_ep}.png', dpi=400)
    plt.show()

    # (b) Generalization Gap
    plt.figure(figsize=(16, 12))
    for delta in delta_list:
        gap_runs = np.array([all_results[local_ep][run][delta]['gap'] for run in range(num_runs)])
        mean_gap = gap_runs.mean(axis=0)[even_indices]
        std_gap  = gap_runs.std(axis=0)[even_indices]
        plt.errorbar(
            even_epochs, mean_gap, yerr=std_gap, capsize=8, elinewidth=4,
            marker='o', markersize=12, label=pretty_label(delta), linewidth=4, alpha=0.85
        )
    plt.title(f'Generalization Gap (local_ep={local_ep})', fontsize=36)
    plt.xlabel('Epoch', fontsize=40)
    plt.ylabel('Generalization Gap', fontsize=40)
    plt.xticks(even_epochs, fontsize=36)
    plt.yticks(fontsize=36)
    plt.legend(loc='best', fontsize=36, frameon=True)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(f'{save_dir}/gap_localep{local_ep}.png', dpi=400)
    plt.show()

# ---------- 4. 每个 δ：不同 local_ep 曲线 ----------
for delta in delta_list:
    # (a) Test Accuracy
    plt.figure(figsize=(16, 12))
    for local_ep in local_ep_list:
        test_acc_runs = np.array([all_results[local_ep][run][delta]['test_acc'] for run in range(num_runs)])
        mean_acc = test_acc_runs.mean(axis=0)[even_indices]
        std_acc  = test_acc_runs.std(axis=0)[even_indices]
        plt.errorbar(
            even_epochs, mean_acc, yerr=std_acc, capsize=8, elinewidth=4,
            marker='o', markersize=12, label=f'local_ep={local_ep}', linewidth=4, alpha=0.85
        )
    plt.title(f'Test Accuracy for {pretty_label(delta)}', fontsize=36)
    plt.xlabel('Epoch', fontsize=40)
    plt.ylabel('Test Accuracy', fontsize=40)
    plt.xticks(even_epochs, fontsize=36)
    plt.yticks([0.6, 0.7, 0.8, 0.9, 1.0], fontsize=36)
    plt.legend(loc='best', fontsize=36, frameon=True)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(f'{save_dir}/test_acc_delta{delta}.png', dpi=400)
    plt.show()

    # (b) Generalization Gap
    plt.figure(figsize=(16, 12))
    for local_ep in local_ep_list:
        gap_runs = np.array([all_results[local_ep][run][delta]['gap'] for run in range(num_runs)])
        mean_gap = gap_runs.mean(axis=0)[even_indices]
        std_gap  = gap_runs.std(axis=0)[even_indices]
        plt.errorbar(
            even_epochs, mean_gap, yerr=std_gap, capsize=8, elinewidth=4,
            marker='o', markersize=12, label=f'local_ep={local_ep}', linewidth=4, alpha=0.85
        )
    plt.title(f'Generalization Gap for {pretty_label(delta)}', fontsize=36)
    plt.xlabel('Epoch', fontsize=40)
    plt.ylabel('Generalization Gap', fontsize=40)
    plt.xticks(even_epochs, fontsize=36)
    plt.yticks(fontsize=36)
    plt.legend(loc='best', fontsize=36, frameon=True)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(f'{save_dir}/gap_delta{delta}.png', dpi=400)
    plt.show()
