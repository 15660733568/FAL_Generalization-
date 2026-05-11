import pickle
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['font.size'] = 18
plt.rcParams['axes.titlesize'] = 30   # 全局标题大小（可选）
plt.rcParams['axes.labelsize'] = 26   # 全局坐标轴标签大小（可选）
plt.rcParams['xtick.labelsize'] = 16
plt.rcParams['ytick.labelsize'] = 16
plt.rcParams['legend.fontsize'] = 24  # 这里默认放大一点
plt.rcParams['figure.titlesize'] = 30
plt.rcParams['mathtext.fontset'] = 'stix'

with open('all_results_fedAT.pkl', 'rb') as f:
    all_results = pickle.load(f)

all_keys = list(all_results.keys())
all_keys_str = [str(k) for k in all_keys]
epsilons = sorted([float(k) for k in all_keys_str])

fraction_labels = {
    0.0: r'$\delta=0$',
    0.01568627450980392: r'$\delta=\frac{4}{255}$',
    0.023529411764705882: r'$\delta=\frac{6}{255}$',
    0.03137254901960784: r'$\delta=\frac{8}{255}$',
}

def pretty_label(eps):
    for k, v in fraction_labels.items():
        if abs(eps - k) < 1e-6:
            return v
    return rf'$\delta={eps}$'

# 只画2,4,...,20这10个点
selected_indices = [i for i in range(1, 20, 2)]  # [1,3,5,...,19]
x_epochs = np.arange(2, 21, 2)

# --- Generalization Gap 曲线 ---
plt.figure(figsize=(10, 9))
gap0 = np.array(all_results[all_keys[0]]['gap'])
num_runs = gap0.shape[0]
for eps in epsilons:
    eps_key = min(all_keys, key=lambda k: abs(float(k) - eps))
    gap_runs = np.array(all_results[eps_key]['gap'])[:, selected_indices]
    mean = np.mean(gap_runs, axis=0)
    std = np.std(gap_runs, axis=0)
    plt.errorbar(
        x_epochs, mean, yerr=std, capsize=5, elinewidth=2.5,
        marker='o', markersize=9, label=pretty_label(eps), linewidth=2.5
    )
plt.title("FAL(SVHN)", fontsize=30)
plt.xlabel("Epoch", fontsize=26)
plt.ylabel("Generalization Gap", fontsize=26)
plt.xticks(x_epochs)
plt.yticks(np.linspace(0, 0.60, 6))
plt.ylim(0, 0.65)
plt.grid(True, linestyle='--', alpha=0.5)
plt.legend(
    loc='center',
    bbox_to_anchor=(0.82, 0.27),
    fontsize=24,      # 放大字体
    markerscale=1.8,  # 放大标记
    frameon=True,
    fancybox=True,
    edgecolor='black',
    framealpha=0.7
)
plt.tight_layout()
plt.savefig('fed_svhn_adv_gap_curve_10pts.png', dpi=400, bbox_inches='tight')
plt.show()

# ---- Test Accuracy 曲线 ----
plt.figure(figsize=(10, 9))
for eps in epsilons:
    eps_key = min(all_keys, key=lambda k: abs(float(k) - eps))
    acc_runs = np.array(all_results[eps_key]['test_acc'])[:, selected_indices]
    mean = np.mean(acc_runs, axis=0)
    std = np.std(acc_runs, axis=0)
    plt.errorbar(
        x_epochs, mean, yerr=std, capsize=5, elinewidth=2.5,
        marker='o', markersize=9, label=pretty_label(eps), linewidth=2.5
    )
plt.title("FAL(SVHN)", fontsize=30)
plt.xlabel("Epoch", fontsize=26)
plt.ylabel("Test Accuracy", fontsize=26)
plt.xticks(x_epochs)
plt.yticks(np.arange(0, 1.01, 0.1))
plt.ylim(0, 1)
plt.grid(True, linestyle='--', alpha=0.5)
plt.legend(
    loc='lower right',
    fontsize=24,      # 放大字体
    markerscale=1.8,  # 放大标记
    frameon=True,
    fancybox=True,
    edgecolor='black',
    framealpha=0.7
)
plt.tight_layout()
plt.savefig('fed_svhn_adv_acc_curve_10pts.png', dpi=400, bbox_inches='tight')
plt.show()
