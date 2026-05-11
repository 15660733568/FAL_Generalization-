import json
import numpy as np
import matplotlib.pyplot as plt
import os

plt.rcParams['font.size'] = 18
plt.rcParams['axes.titlesize'] = 30
plt.rcParams['axes.labelsize'] = 26
plt.rcParams['xtick.labelsize'] = 16
plt.rcParams['ytick.labelsize'] = 16
plt.rcParams['legend.fontsize'] = 24
plt.rcParams['figure.titlesize'] = 30
plt.rcParams['mathtext.fontset'] = 'stix'

# ========== 路径和数据读取 ==========
json_file = 'D:/pyproject/fed/results/adv_webspam_cnn_epochs20_C[0.2]_iid[2]_simple.json'
with open(json_file, 'r', encoding='utf-8') as f:
    saved_data = json.load(f)
all_results = saved_data['all_results']

# ========== fraction_labels 和 pretty_label ==========
fraction_labels = {
    0.0: r'$\delta=0$',
    0.1: r'$\delta=0.1$',
    0.2: r'$\delta=0.2$',
    0.3: r'$\delta=0.3$',
}
def pretty_label(eps):
    for k, v in fraction_labels.items():
        if abs(float(eps) - k) < 1e-6:
            return v
    return rf'$\delta={eps}$'

# ========== 配置参数 ==========
epochs_num = 20
dataset = 'webspam'
model = 'cnn'
frac = 0.2
iid = 1
selected_indices = [i for i in range(1, 20, 2)]  # [1,3,5,...,19]
x_epochs = np.arange(2, 21, 2)
epsilons = sorted([float(k) for k in all_results.keys()])
all_keys = sorted(all_results.keys(), key=lambda x: float(x))

save_dir = os.path.dirname(json_file)
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

# ========== Generalization Gap 曲线 ==========
plt.figure(figsize=(10, 9))
for eps in epsilons:
    eps_str = str(eps)
    gap_runs = np.array([np.array(run['gen_gap'])[selected_indices] for run in all_results[eps_str]])
    mean = np.mean(gap_runs, axis=0)
    std = np.std(gap_runs, axis=0)
    plt.errorbar(
        x_epochs, mean, yerr=std, capsize=5, elinewidth=2.5,
        marker='o', markersize=9, label=pretty_label(eps), linewidth=2.5
    )
plt.title("FAL(webspam)", fontsize=30)
plt.xlabel("Epoch", fontsize=26)
plt.ylabel("Generalization Gap", fontsize=26)
plt.xticks(x_epochs)
plt.ylim(0, 0.14)
plt.grid(True, linestyle='--', alpha=0.5)
plt.legend(
    loc='center',
    bbox_to_anchor=(0.8, 0.2),   # 关键！浮在两根线之间的空白处
    fontsize=24,
    markerscale=1.8,
    frameon=True,
    fancybox=True,
    edgecolor='black',
    framealpha=0.7
)
plt.tight_layout()
gap_path = os.path.join(save_dir, f'adv_gap_errorbar_{dataset}_{model}_epochs{epochs_num}_C[{frac}]_iid[{iid}_10pts].png')
plt.savefig(gap_path, dpi=400, bbox_inches='tight')
plt.show()
print(f'已保存Gen Gap误差棒曲线图: {gap_path}')

# ========== Test Accuracy 曲线 ==========
plt.figure(figsize=(10, 9))
for eps in epsilons:
    eps_str = str(eps)
    acc_runs = np.array([np.array(run['adv_acc'])[selected_indices] for run in all_results[eps_str]])
    mean = np.mean(acc_runs, axis=0)
    std = np.std(acc_runs, axis=0)
    plt.errorbar(
        x_epochs, mean, yerr=std, capsize=5, elinewidth=2.5,
        marker='o', markersize=9, label=pretty_label(eps), linewidth=2.5
    )
plt.title("FAL(webspam)", fontsize=30)
plt.xlabel("Epoch", fontsize=26)
plt.ylabel("Test Accuracy", fontsize=26)
plt.xticks(x_epochs)
plt.ylim(0.8, 1)  # 这里设置纵轴范围
plt.grid(True, linestyle='--', alpha=0.5)
plt.legend(
    loc='lower right',
    fontsize=24,
    markerscale=1.8,
    frameon=True,
    fancybox=True,
    edgecolor='black',
    framealpha=0.7
)
plt.tight_layout()
acc_path = os.path.join(save_dir, f'adv_acc_errorbar_{dataset}_{model}_epochs{epochs_num}_C[{frac}]_iid[{iid}_10pts].png')
plt.savefig(acc_path, dpi=400, bbox_inches='tight')
plt.show()
print(f'已保存Test Accuracy误差棒曲线图: {acc_path}')
