# FAL_SUSY.py
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

from options import args_parser
from update import LocalUpdate, test_inference, adversarial_test
from utils import get_dataset, average_weights, exp_details
from model import CNNSUSY

def eval_clean(args, model, dataset):
    return test_inference(args, model, dataset)[0]

def eval_adv(model, dataset, eps, device, iters=10, alpha=None):
    return adversarial_test(model, dataset, epsilon=eps, device=device, iters=iters, alpha=alpha, norm='Linf')

def _build_model(args, train_dataset):
    # 更稳：根据样本推断输入维度
    try:
        sample_x, _ = train_dataset[0]
        input_dim = int(np.prod(sample_x.shape))
    except Exception:
        input_dim = 18
    return CNNSUSY(num_classes=2, input_dim=input_dim)

def run_federated_training(args, epsilon):
    logger = SummaryWriter('../logs')
    device = 'cuda' if (isinstance(args.gpu, int) and args.gpu >= 0 and torch.cuda.is_available()) else 'cpu'
    if device == 'cuda':
        torch.cuda.set_device(args.gpu)

    # 数据集与客户端划分
    train_dataset, test_dataset, user_groups = get_dataset(args)

    # 模型
    global_model = _build_model(args, train_dataset).to(device)
    global_model.train()
    global_weights = global_model.state_dict()

    current_lr = args.lr
    train_loss_curve = []
    gap_curve = []
    clean_test_curve, adv_test_curve = [], []
    clean_train_curve, adv_train_curve = [], []

    print(f"\n========== SUSY Federated Training (epsilon={epsilon}) ==========")

    for epoch in tqdm(range(args.epochs)):
        if (epoch + 1) % args.lr_decay_epoch == 0:
            current_lr *= args.lr_decay

        local_weights, local_losses = [], []
        m = max(int(args.frac * args.num_users), 1)
        idxs_users = np.random.choice(range(args.num_users), m, replace=False)

        for idx in idxs_users:
            local_model = LocalUpdate(args=args, dataset=train_dataset, idxs=user_groups[idx], logger=logger)
            w, loss = local_model.update_weights(model=copy.deepcopy(global_model),
                                                global_round=epoch,
                                                current_lr=current_lr)
            local_weights.append(copy.deepcopy(w))
            local_losses.append(copy.deepcopy(loss))

        # FedAvg
        global_weights = average_weights(local_weights)
        global_model.load_state_dict(global_weights)

        # 平均 loss
        loss_avg = float(sum(local_losses) / len(local_losses))
        train_loss_curve.append(loss_avg)

        # --- 全局评测 ---
        device_flag = 'cuda' if (isinstance(args.gpu, int) and args.gpu >= 0 and torch.cuda.is_available()) else 'cpu'

        if epsilon <= 0.0:
            # 非对抗：train/test 都是干净样本
            train_acc_clean, _ = test_inference(args, global_model, train_dataset)
            test_acc_clean, _  = test_inference(args, global_model, test_dataset)
            gap = train_acc_clean - test_acc_clean

            clean_train_curve.append(train_acc_clean)
            clean_test_curve.append(test_acc_clean)
            adv_train_curve.append(train_acc_clean)  # 对齐长度
            adv_test_curve.append(test_acc_clean)
        else:
            # 纯对抗：train/test 都是对抗样本
            alpha = epsilon / 4.0
            train_acc_adv = eval_adv(global_model, train_dataset, eps=epsilon, device=device_flag, iters=10, alpha=alpha)
            test_acc_adv  = eval_adv(global_model, test_dataset,  eps=epsilon, device=device_flag, iters=10, alpha=alpha)
            gap = train_acc_adv - test_acc_adv

            adv_train_curve.append(train_acc_adv)
            adv_test_curve.append(test_acc_adv)
            # 同时计算干净（仅作参考，不计入 gap）
            train_acc_clean, _ = test_inference(args, global_model, train_dataset)
            test_acc_clean, _  = test_inference(args, global_model, test_dataset)
            clean_train_curve.append(train_acc_clean)
            clean_test_curve.append(test_acc_clean)

        gap_curve.append(gap)

        # 只打印全局（不打印每个客户端）
        if epsilon <= 0.0:
            print(f"Epoch {epoch + 1:03d} | Loss={loss_avg:.4f} | "
                  f"TrainClean={clean_train_curve[-1]:.4f} | TestClean={clean_test_curve[-1]:.4f} | Gap={gap:.4f}")
        else:
            print(f"Epoch {epoch + 1:03d} | Loss={loss_avg:.4f} | "
                  f"TrainAdv={adv_train_curve[-1]:.4f} | TestAdv={adv_test_curve[-1]:.4f} | Gap={gap:.4f}")

    results = {
        'epsilon': epsilon,
        'loss': train_loss_curve,
        'gap': gap_curve,
        'clean_train': clean_train_curve,
        'clean_test': clean_test_curve,
        'adv_train': adv_train_curve,
        'adv_test': adv_test_curve,
    }

    return global_model, results

def main():
    start_time = time.time()
    args = args_parser()
    exp_details(args)

    EPSILONS = [ 0,4/255]

    all_runs = {}
    for eps in EPSILONS:
        # 切换对抗模式
        args.epsilon = float(eps)
        args.alpha = float(eps) / 4.0 if eps > 0 else 0.0
        args.adv_train = 1 if eps > 0 else 0
        args.attack_iters = 10
        args.attack_norm = 'Linf'

        model, results = run_federated_training(args, epsilon=eps)

        # 保存
        ckpt = f'susy_fed_adv_eps{eps:.6f}.pt'
        torch.save(model.state_dict(), ckpt)
        pkl_path = f'susy_fed_results_eps{eps:.6f}.pkl'
        with open(pkl_path, 'wb') as f:
            pickle.dump(results, f)

        all_runs[eps] = results
        print(f"\n[Saved] Model -> {ckpt}\n[Saved] Results -> {pkl_path}\n")

    print("\n=========== Summary ===========")
    for eps, res in all_runs.items():
        if eps <= 0.0:
            print(f"Eps={eps:.6f} | Final TrainClean={res['clean_train'][-1]:.4f} "
                  f"| TestClean={res['clean_test'][-1]:.4f} | Gap={res['gap'][-1]:.4f}")
        else:
            print(f"Eps={eps:.6f} | Final TrainAdv={res['adv_train'][-1]:.4f} "
                  f"| TestAdv={res['adv_test'][-1]:.4f} | Gap={res['gap'][-1]:.4f}")

    print(f"\nTotal Runtime: {(time.time() - start_time):.2f} sec")

if __name__ == '__main__':
    main()
