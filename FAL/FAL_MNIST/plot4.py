import pickle
import numpy as np
import matplotlib.pyplot as plt

# ---------- 1. 全局绘图字号 ----------
plt.rcParams['font.size']        = 34
plt.rcParams['axes.titlesize']   = 38
plt.rcParams['axes.labelsize']   = 38
plt.rcParams['xtick.labelsize']  = 34
plt.rcParams['ytick.labelsize']  = 34
plt.rcParams['legend.fontsize']  = 36
plt.rcParams['figure.titlesize'] = 42
plt.rcParams['mathtext.fontset'] = 'stix'  # LaTeX 风格字体

# ---------- 2. 载入结果 ----------
with open('save/adv_results_all_runs.pkl', 'rb') as f:   # ← 若路径不同，改这里
    all_results = pickle.load(f)

epsilons    = [0.0, 0.1, 0.3, 0.5]
num_runs    = len(all_results)
num_epochs  = len(all_results[0][epsilons[0]]['test_acc'])   # 原始 20 个 epoch

# 只保留偶数 epoch：2,4,…,20
even_epochs  = list(range(2, num_epochs + 1, 2))   # 横轴刻度
even_indices = np.arange(1, num_epochs, 2)         # 对应的 zero-based 索引 1,3,…,19

def pretty_label(eps):
    return rf'$\delta={eps}$'

# ---------- 3. 测试准确率 ----------
plt.figure(figsize=(16, 12))
for eps in epsilons:
    runs = np.array([run[eps]['test_acc'] for run in all_results])
    mean = runs.mean(axis=0)[even_indices]
    std  = runs.std(axis=0)[even_indices]
    plt.errorbar(
        even_epochs, mean, yerr=std, capsize=8, elinewidth=4,
        marker='o', markersize=12, label=pretty_label(eps), linewidth=4
    )

plt.title("FAL")
plt.xlabel("Epoch")
plt.ylabel("Test Accuracy")
plt.xticks(even_epochs)
plt.ylim(0.7, 1.0)
plt.yticks([0.7, 0.8, 0.9, 1.0])
plt.legend(loc='lower right', frameon=True)   # ← 放到右下角
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.savefig('test_acc_curve_even.png', dpi=400)
plt.show()


# ---------- 4. 泛化 Gap ----------
plt.figure(figsize=(16, 12))
for eps in epsilons:
    runs = np.array([run[eps]['gap'] for run in all_results])  # shape: (runs, 20)
    mean = runs.mean(axis=0)[even_indices]
    std  = runs.std(axis=0)[even_indices]
    plt.errorbar(
        even_epochs, mean, yerr=std, capsize=8, elinewidth=4,
        marker='o', markersize=12, label=pretty_label(eps), linewidth=4
    )

plt.title("Generalization Gap over Epochs (mean±std)")
plt.xlabel("Epoch")
plt.ylabel("Generalization Gap")
plt.xticks(even_epochs)
plt.legend(frameon=True)
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.savefig('gen_gap_curve_even.png', dpi=400)
plt.show()
