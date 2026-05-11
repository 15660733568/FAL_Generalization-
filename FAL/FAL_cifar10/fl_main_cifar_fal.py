#!/usr/bin/env python
# -*- coding: utf-8 -*-
# CIFAR federated learning main following the FAL_SVHN-style pipeline
# Includes adversarial training (PGD) and adversarial evaluation.
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

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
from model_cifar import CNNCifar

def run_once(args):
    """Run one full FL training with (optional) adversarial training enabled in LocalUpdate."""
    logger = SummaryWriter('../logs')
    if args.gpu:
        torch.cuda.set_device(args.gpu)
    device = 'cuda' if args.gpu else 'cpu'

    # load dataset and user groups
    train_dataset, test_dataset, user_groups = get_dataset(args)

    # build model
    if args.model == 'cnn':
        global_model = CNNCifar()
    else:
        raise ValueError('This entry script currently supports model=cnn with CIFAR10.')

    global_model.to(device)
    global_model.train()

    # copy initial weights
    # (Algorithm 1: initialize w0 at server)
    global_weights = global_model.state_dict()

    train_acc_curve, test_acc_curve, gen_gap_curve = [], [], []
    train_loss_curve = []

    current_lr = args.lr

    for epoch in tqdm(range(args.epochs)):
        # learning rate decay (same as your CIFAR main)
        if (epoch + 1) % args.lr_decay_epoch == 0:
            current_lr *= args.lr_decay
            print(f"\nLearning rate decayed to: {current_lr}\n")

        local_weights, local_losses = [], []
        print(f"\n | Global Training Round : {epoch + 1} |\n")

        global_model.train()
        m = max(int(args.frac * args.num_users), 1)
        idxs_users = np.random.choice(range(args.num_users), m, replace=False)

        for idx in idxs_users:
            local_model = LocalUpdate(args=args, dataset=train_dataset,
                                      idxs=user_groups[idx], logger=logger)
            w, loss = local_model.update_weights(
                model=copy.deepcopy(global_model),
                global_round=epoch,
                current_lr=current_lr)
            local_weights.append(copy.deepcopy(w))
            local_losses.append(copy.deepcopy(loss))

        # FedAvg
        global_weights = average_weights(local_weights)
        global_model.load_state_dict(global_weights)

        # Curves
        loss_avg = float(sum(local_losses) / len(local_losses))
        train_loss_curve.append(loss_avg)

        # Evaluate on train/test (clean)
        train_acc, _ = test_inference(args, global_model, train_dataset)
        test_acc, _ = test_inference(args, global_model, test_dataset)
        gap = train_acc - test_acc

        train_acc_curve.append(train_acc)
        test_acc_curve.append(test_acc)
        gen_gap_curve.append(gap)

        print(f"Epoch {epoch + 1}: Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}, Gen Gap: {gap:.4f}")

    return global_model, (train_loss_curve, train_acc_curve, test_acc_curve, gen_gap_curve), (train_dataset, test_dataset)

def main():
    start_time = time.time()

    args = args_parser()
    exp_details(args)

    # FAL-style experimental plan: sweep several epsilons and repeat runs.
    epsilons = [0,4/255]
    num_repeats = 5

    # To store all metrics across epsilons & repeats
    all_results = {
        eps: {
            'train_acc': [],
            'test_acc': [],
            'gen_gap': [],
            'adv_test_acc': {test_eps: [] for test_eps in epsilons}
        } for eps in epsilons
    }

    for epsilon in epsilons:
        # enable adversarial training in LocalUpdate via args
        args.adv_train = 1
        args.epsilon = float(epsilon)
        # heuristic PGD step
        args.alpha = float(epsilon) / 4.0
        args.attack_iters = 10
        args.restarts = 1
        args.attack_norm = 'Linf'

        for r in range(num_repeats):
            print(f"\n=== CIFAR10 FAL-style run | adv ε={epsilon:.6f} | repeat {r+1}/{num_repeats} ===")
            global_model, curves, datasets = run_once(args)
            train_loss_curve, train_acc_curve, test_acc_curve, gen_gap_curve = curves
            train_dataset, test_dataset = datasets

            # save curves
            all_results[epsilon]['train_acc'].append(train_acc_curve)
            all_results[epsilon]['test_acc'].append(test_acc_curve)
            all_results[epsilon]['gen_gap'].append(gen_gap_curve)

            # Adversarial evaluation at multiple eps test strengths
            from torch.utils.data import DataLoader
            test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)
            device = 'cuda' if args.gpu else 'cpu'
            for test_eps in epsilons:
                adv_acc = adversarial_test(global_model, test_loader, test_eps, device=device, iters=20, alpha=test_eps/4.0)
                all_results[epsilon]['adv_test_acc'][test_eps].append(adv_acc)
                print(f"Adversarial Test @ ε={test_eps:.6f}: acc={adv_acc:.4f}")

            # Save model checkpoint
            ckpt = f"cifar_fal_adv_eps{epsilon:.6f}_rep{r}.pt"
            torch.save(global_model.state_dict(), ckpt)
            print(f"Saved: {ckpt}")

    # Persist results
    with open('cifar_fal_all_results.pkl', 'wb') as f:
        pickle.dump(all_results, f)

    # Print summary
    print("\nFinal summary across epsilons:")
    for eps in epsilons:
        last_tests = [accs[-1] for accs in all_results[eps]['test_acc']]
        last_gaps  = [gaps[-1] for gaps in all_results[eps]['gen_gap']]
        print(f"ε={eps:.6f} | Test Acc (final): {np.mean(last_tests):.4f} ± {np.std(last_tests):.4f} | Gap: {np.mean(last_gaps):.4f} ± {np.std(last_gaps):.4f}")
        for teps in epsilons:
            adv_list = all_results[eps]['adv_test_acc'][teps]
            print(f"  Adv eval ε={teps:.6f}: {np.mean(adv_list):.4f} ± {np.std(adv_list):.4f}")

    print('\nTotal Run Time: {0:0.4f}'.format(time.time() - start_time))

if __name__ == '__main__':
    main()
