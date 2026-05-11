import pickle
import numpy as np
import matplotlib.pyplot as plt

# 更大字体和点线
plt.rcParams['font.size'] = 34
plt.rcParams['axes.titlesize'] = 38
plt.rcParams['axes.labelsize'] = 38
plt.rcParams['xtick.labelsize'] = 34
plt.rcParams['ytick.labelsize'] = 34
plt.rcParams['legend.fontsize'] = 36
plt.rcParams['figure.titlesize'] = 42
plt.rcParams['mathtext.fontset'] = 'stix'  # 和latex论文字体一致

with open('save/adv_results_all_runs.pkl', 'rb') as f:
    all_results = pickle.load(f)

epsilons = [0.0, 0.1, 0.3, 0.5]
num_runs = len(all_results)
num_epochs = len(all_results[0][epsilons[0]]['test_acc'])

def pretty_label(eps):
    return rf'$\delta={eps}$'

# (1) 测试准确率
plt.figure(figsize=(16, 12))
for i, eps in enumerate(epsilons):
    test_acc_runs = np.array([run[eps]['test_acc'] for run in all_results])
    mean = np.mean(test_acc_runs, axis=0)
    std = np.std(test_acc_runs, axis=0)
    plt.errorbar(
        range(1, num_epochs+1), mean, yerr=std, capsize=8, elinewidth=4,
        marker='o', markersize=12, label=pretty_label(eps), linewidth=4
    )
plt.title("FAL", fontsize=36)
plt.xlabel("Epoch", fontsize=40)
plt.ylabel("Test Accuracy", fontsize=40)
plt.xticks(range(1, num_epochs+1), fontsize=36)
plt.ylim(0.6, 1.0)  # y轴范围从0.7到1.0
plt.yticks([0.6,0.7, 0.8, 0.9,1.0], fontsize=36)  # 只显示四个刻度
plt.legend(loc='best', fontsize=36, frameon=True)
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.savefig('save/test_acc_curve_full.png', dpi=400)
plt.show()
# (2) 泛化gap
plt.figure(figsize=(16, 12))
for i, eps in enumerate(epsilons):
    gap_runs = np.array([run[eps]['gap'] for run in all_results])
    mean = np.mean(gap_runs, axis=0)
    std = np.std(gap_runs, axis=0)
    plt.errorbar(
        range(1, num_epochs+1), mean, yerr=std, capsize=8, elinewidth=4,
        marker='o', markersize=12, label=pretty_label(eps), linewidth=4
    )
plt.title("Generalization Gap over Epochs (mean±std)", fontsize=36)
plt.xlabel("Epoch", fontsize=36)
plt.ylabel("Generalization Gap", fontsize=36)
plt.xticks(range(1, num_epochs+1), fontsize=34)
plt.yticks(fontsize=34)
plt.legend(loc='best', fontsize=36, frameon=True)
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.savefig('save/gen_gap_curve_full.png', dpi=400)
plt.show()
