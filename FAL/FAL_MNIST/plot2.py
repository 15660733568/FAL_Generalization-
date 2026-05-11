import pickle
import numpy as np
import matplotlib.pyplot as plt
import os

# 全局字体和风格设置
plt.rcParams['font.size'] = 18
plt.rcParams['axes.titlesize'] = 24
plt.rcParams['axes.labelsize'] = 22
plt.rcParams['xtick.labelsize'] = 20
plt.rcParams['ytick.labelsize'] = 20
plt.rcParams['legend.fontsize'] = 24
plt.rcParams['figure.titlesize'] = 24
plt.rcParams['mathtext.fontset'] = 'stix'

def pretty_label(delta):
    return rf'$\delta={delta}$'

# 加载数据
with open('save/adv_results_all_runs_M.pkl', 'rb') as f:
    all_results = pickle.load(f)

num_users_list = [20, 50, 100]
deltas = [0.0, 0.1, 0.3, 0.5]
num_runs = 3  # 你的实验重复次数
colors = ['#1f77b4', '#2ca02c', '#d62728', '#9467bd']
markers = ['o', 's', '^', 'D']
save_dir = 'save'
os.makedirs(save_dir, exist_ok=True)

# 只取epoch 2,4,6,...,20（即索引1,3,...,19），共10个点
selected_indices = [i for i in range(1, 20, 2)]  # 1,3,5,...,19
x_epochs = np.arange(2, 21, 2)                   # 2,4,...,20

# 针对每个 delta 画 num_users 对比图
for d_idx, delta in enumerate(deltas):
    plt.figure(figsize=(10, 8))
    for n_idx, num_users in enumerate(num_users_list):
        run_test_accs = []
        run_gaps = []
        for run_results in all_results[num_users]:
            run_test_accs.append(np.array(run_results[delta]['test_acc'])[selected_indices])
            run_gaps.append(np.array(run_results[delta]['gap'])[selected_indices])
        run_test_accs = np.array(run_test_accs)
        run_gaps = np.array(run_gaps)
        mean_acc = np.mean(run_test_accs, axis=0)
        std_acc = np.std(run_test_accs, axis=0)
        mean_gap = np.mean(run_gaps, axis=0)
        std_gap = np.std(run_gaps, axis=0)
        plt.errorbar(
            x_epochs, mean_acc, yerr=std_acc,
            label=f'num_users={num_users}', color=colors[n_idx],
            marker=markers[n_idx], capsize=3, linewidth=2, markersize=7, elinewidth=1.5, alpha=0.85)
    plt.title(f"Test Accuracy Over Epochs ({pretty_label(delta)})")
    plt.xlabel("Epoch")
    plt.ylabel("Test Accuracy")
    plt.xticks(x_epochs)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(f'{save_dir}/test_acc_curve_delta_{delta}_10pts.png', dpi=300)
    plt.show()

    # 画gap
    plt.figure(figsize=(10, 8))
    for n_idx, num_users in enumerate(num_users_list):
        run_gaps = [np.array(run_results[delta]['gap'])[selected_indices] for run_results in all_results[num_users]]
        run_gaps = np.array(run_gaps)
        mean_gap = np.mean(run_gaps, axis=0)
        std_gap = np.std(run_gaps, axis=0)
        plt.errorbar(
            x_epochs, mean_gap, yerr=std_gap,
            label=f'num_users={num_users}', color=colors[n_idx],
            marker=markers[n_idx], capsize=3, linewidth=2, markersize=7, elinewidth=1.5, alpha=0.85)
    plt.title(f"Generalization Gap Over Epochs ({pretty_label(delta)})")
    plt.xlabel("Epoch")
    plt.ylabel("Generalization Gap")
    plt.xticks(x_epochs)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(f'{save_dir}/gen_gap_curve_delta_{delta}_10pts.png', dpi=300)
    plt.show()

# 对比每个num_users下，不同delta曲线
for n_idx, num_users in enumerate(num_users_list):
    plt.figure(figsize=(10, 8))
    for d_idx, delta in enumerate(deltas):
        run_test_accs = [np.array(run_results[delta]['test_acc'])[selected_indices] for run_results in all_results[num_users]]
        run_test_accs = np.array(run_test_accs)
        mean_acc = np.mean(run_test_accs, axis=0)
        std_acc = np.std(run_test_accs, axis=0)
        plt.errorbar(
            x_epochs, mean_acc, yerr=std_acc,
            label=pretty_label(delta), color=colors[d_idx],
            marker=markers[d_idx], capsize=3, linewidth=2, markersize=7, elinewidth=1.5, alpha=0.85)
    plt.title(f"Test Accuracy Over Epochs (num_users={num_users})")
    plt.xlabel("Epoch")
    plt.ylabel("Test Accuracy")
    plt.xticks(x_epochs)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(f'{save_dir}/test_acc_curve_numusers_{num_users}_10pts.png', dpi=300)
    plt.show()

    # gap
    plt.figure(figsize=(10, 8))
    for d_idx, delta in enumerate(deltas):
        run_gaps = [np.array(run_results[delta]['gap'])[selected_indices] for run_results in all_results[num_users]]
        run_gaps = np.array(run_gaps)
        mean_gap = np.mean(run_gaps, axis=0)
        std_gap = np.std(run_gaps, axis=0)
        plt.errorbar(
            x_epochs, mean_gap, yerr=std_gap,
            label=pretty_label(delta), color=colors[d_idx],
            marker=markers[d_idx], capsize=3, linewidth=2, markersize=7, elinewidth=1.5, alpha=0.85)
    plt.title(f"Generalization Gap Over Epochs (num_users={num_users})")
    plt.xlabel("Epoch")
    plt.ylabel("Generalization Gap")
    plt.xticks(x_epochs)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(f'{save_dir}/gen_gap_curve_numusers_{num_users}_10pts.png', dpi=300)
    plt.show()
