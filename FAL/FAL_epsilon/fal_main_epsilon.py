#!/usr/bin/env python
# -*- coding: utf-8 -*-
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
from update import LocalUpdate, test_inference
from models import MLP, CNNMnist, CNNFashion_Mnist, ImprovedCNN, CNNWebspam, CNN1D
from utils import get_dataset, average_weights, exp_details
import matplotlib.pyplot as plt
from collections import defaultdict
from attacks import PGDAttack

def plot_results(all_results, epsilons, args):
    epochs = list(range(args.epochs))
    plt.figure(figsize=(15, 6))
    plt.subplot(1, 2, 1)
    for epsilon in epsilons:
        gen_gaps = np.array([run['gen_gap'] for run in all_results[epsilon]])
        mean_gap = np.mean(gen_gaps, axis=0)
        std_gap = np.std(gen_gaps, axis=0)
        plt.plot(epochs, mean_gap, label=f'ε={epsilon}', linewidth=2)
        plt.fill_between(epochs, mean_gap-std_gap, mean_gap+std_gap, alpha=0.2)
    plt.xlabel('Communication Rounds', fontsize=12)
    plt.ylabel('Generalization Gap', fontsize=12)
    plt.title('Generalization Gap vs Communication Rounds', fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(True)
    plt.subplot(1, 2, 2)
    for epsilon in epsilons:
        test_accs = np.array([run['test_acc'] for run in all_results[epsilon]])
        mean_acc = np.mean(test_accs, axis=0) * 100
        std_acc = np.std(test_accs, axis=0) * 100
        plt.plot(epochs, mean_acc, label=f'ε={epsilon}', linewidth=2)
        plt.fill_between(epochs, mean_acc-std_acc, mean_acc+std_acc, alpha=0.2)
    plt.xlabel('Communication Rounds', fontsize=12)
    plt.ylabel('Test Accuracy (%)', fontsize=12)
    plt.title('Test Accuracy vs Communication Rounds', fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(True)
    plt.tight_layout()
    if not os.path.exists('../fed/plots'):
        os.makedirs('../fed/plots')
    plot_filename = f'../fed/plots/adv_{args.dataset}_{args.model}_epochs{args.epochs}_C[{args.frac}]_iid[{args.iid}].png'
    plt.savefig(plot_filename)
    plt.close()
    print(f'\nSaved plots to {plot_filename}')

def eval_adv_acc(args, model, test_dataset, epsilon):
    device = args.device
    model.eval()
    if epsilon == 0:
        correct = 0
        total = 0
        testloader = torch.utils.data.DataLoader(test_dataset, batch_size=128, shuffle=False)
        for images, labels in testloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = outputs.max(1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
        acc = correct / total
        return acc
    else:
        attack = PGDAttack(model, epsilon=epsilon,
                        alpha=args.alpha,
                        iters=args.attack_iters,
                        restarts=args.restarts,
                        norm=args.attack_norm,
                        num_classes=args.num_classes)
        correct = 0
        total = 0
        testloader = torch.utils.data.DataLoader(test_dataset, batch_size=128, shuffle=False)
        for images, labels in testloader:
            images, labels = images.to(device), labels.to(device)
            adv_images = attack.perturb(images, labels)
            outputs = model(adv_images)
            _, predicted = outputs.max(1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
        acc = correct / total
        return acc

def main_train(args, run_id=None, epsilon=None):
    start_time = time.time()
    logger = SummaryWriter('../logs')
    train_dataset, test_dataset, user_groups = get_dataset(args)

    # 构建模型
    if args.model == 'cnn':
        if args.dataset == 'mnist':
            global_model = CNNMnist(args=args)
        elif args.dataset == 'fmnist':
            global_model = CNNFashion_Mnist(args=args)
        elif args.dataset == 'svhn':
            global_model = ImprovedCNN(num_classes=10)
        elif args.dataset == 'epsilon':
            input_dim = train_dataset[0][0].shape[0]
            global_model = CNN1D(input_dim=input_dim, num_classes=args.num_classes)
    elif args.model == 'mlp':
        img_size = train_dataset[0][0].shape
        len_in = 1
        for x in img_size:
            len_in *= x
        global_model = MLP(dim_in=len_in, dim_hidden=64, dim_out=args.num_classes)
    else:
        exit('Error: unrecognized model')

    global_model.to(args.device)
    global_model.train()
    # print("\nModel Architecture:")
    # print(global_model)

    train_loss, train_accuracy = [], []
    gen_gaps, test_accuracies = [], []
    current_lr = args.lr

    for epoch in tqdm(range(args.epochs)):
        if (epoch + 1) % args.lr_decay_epoch == 0:
            current_lr *= args.lr_decay
            print(f'\nLearning rate decayed to: {current_lr}\n')

        local_weights, local_losses = [], []

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

        global_weights = average_weights(local_weights)
        global_model.load_state_dict(global_weights)
        loss_avg = sum(local_losses) / len(local_losses)
        train_loss.append(loss_avg)

        global_model.eval()
        train_acc, _ = test_inference(args, global_model, train_dataset)
        adv_acc = eval_adv_acc(args, global_model, test_dataset, epsilon=args.epsilon)
        gen_gap = abs(train_acc - adv_acc)
        train_accuracy.append(train_acc)
        test_accuracies.append(adv_acc)
        gen_gaps.append(gen_gap)

        print(f"[Global Round {epoch + 1}/{args.epochs}] "
              f"Train Acc: {train_acc:.4f} | Adv Test Acc: {adv_acc:.4f} | Generalization Gap: {gen_gap:.4f}")

    final_gen_gap = abs(train_accuracy[-1] - test_accuracies[-1])
    print(f'\nResults after {args.epochs} global rounds of training:')
    print(f"|---- Final Train Accuracy: {100 * train_accuracy[-1]:.2f}%")
    print(f"|---- Final Adv Test Accuracy: {100 * test_accuracies[-1]:.2f}%")
    print(f"|---- Final Generalization Gap (|Train Acc - Adv Test Acc|): {final_gen_gap:.4f}")
    print('\nTotal Run Time: {0:0.4f} seconds'.format(time.time() - start_time))

    if epsilon is not None and run_id is not None:
        save_dir = '../Fal_epsilon/save/individual'
        os.makedirs(save_dir, exist_ok=True)
        pkl_path = os.path.join(
            save_dir, f'{args.dataset}_{args.model}_eps{epsilon}_run{run_id}.pkl')
        with open(pkl_path, 'wb') as f:
            pickle.dump({
                'train_loss': train_loss,
                'train_acc': train_accuracy,
                'adv_acc': test_accuracies,
                'gen_gap': gen_gaps,
                'final_adv_acc': test_accuracies[-1],
                'final_gen_gap': final_gen_gap,
                'model': global_model.state_dict(),
                'args': args,
                'epsilon': epsilon,
                'run': run_id
            }, f)
        print(f'\n[Saved experiment result to {pkl_path}]')

    return {
        'train_loss': train_loss,
        'train_acc': train_accuracy,
        'adv_acc': test_accuracies,
        'gen_gap': gen_gaps,
        'final_adv_acc': test_accuracies[-1],
        'final_gen_gap': final_gen_gap,
        'model': global_model.state_dict()
    }

if __name__ == '__main__':
    args = args_parser()
    assert torch.cuda.is_available(), "CUDA not available! Please use a GPU machine."
    args.device = 'cuda'
    args.print_every = 1
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    epsilons = [0.0,0.005, 0.01, 0.02] if args.dataset == 'epsilon' else [0.01, 0.03, 0.1]
    num_runs = 5

    all_results = defaultdict(list)
    trained_models = {}

    for epsilon in epsilons:
        print(f"\n=== Starting experiments with epsilon={epsilon} ===")
        args.epsilon = epsilon

        for run in range(num_runs):
            print(f"\n--- Run {run+1}/{num_runs} ---")
            run_result = main_train(args, run_id=run, epsilon=epsilon)
            all_results[epsilon].append(run_result)
            model_key = f"eps_{epsilon}_run_{run}"
            trained_models[model_key] = run_result['model']

            # ========== 这里是新增的保存每个模型每次实验的代码 ==========
            save_root = '../fed/results/individual'
            os.makedirs(save_root, exist_ok=True)
            pkl_path = os.path.join(
                save_root, f'{args.dataset}_{args.model}_eps{epsilon}_run{run}.pkl')
            with open(pkl_path, 'wb') as f:
                pickle.dump(run_result, f)
            print(f'[Saved single experiment result to {pkl_path}]')
            # ========================================================

    save_root = '../fed/results'
    os.makedirs(save_root, exist_ok=True)
    results_filename = f'{save_root}/adv_{args.dataset}_{args.model}_epochs{args.epochs}_C[{args.frac}]_iid[{args.iid}].pkl'
    with open(results_filename, 'wb') as f:
        pickle.dump({
            'all_results': all_results,
            'trained_models': trained_models,
            'args': args
        }, f)
    print(f'\nSaved all results to {results_filename}')

    plot_results(all_results, epsilons, args)

    print("\n=== Final Summary ===")
    for epsilon in epsilons:
        final_adv_accs = [run['final_adv_acc'] for run in all_results[epsilon]]
        final_gen_gaps = [run['final_gen_gap'] for run in all_results[epsilon]]
        avg_acc = np.mean(final_adv_accs) * 100
        std_acc = np.std(final_adv_accs) * 100
        avg_gap = np.mean(final_gen_gaps)
        std_gap = np.std(final_gen_gaps)
        print(f"ε={epsilon}: Adv Test Accuracy = {avg_acc:.2f}% ± {std_acc:.2f}% | Final Gen Gap = {avg_gap:.4f} ± {std_gap:.4f}")
