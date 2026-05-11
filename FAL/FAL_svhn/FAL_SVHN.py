import os
import copy
import time
import torch
import pickle
import numpy as np
import attack_generator as attack
from models import *
from tqdm import tqdm
from logger import Logger
from options import args_parser
from matplotlib.pyplot import title
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
from update import LocalUpdate, test_inference
from utils import get_dataset, average_weights, exp_details, average_weights_alpha

def save_checkpoint(state, checkpoint='../FAT_result', filename='checkpoint.pth.tar'):
    filepath = os.path.join(checkpoint, filename)
    torch.save(state, filepath)

if __name__ == '__main__':
    start_time = time.time()
    logger = SummaryWriter('../logs')
    args = args_parser()
    exp_details(args)

    seed = args.seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True

    # Set output dir
    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir)

    device = 'cuda' if args.gpu else 'cpu'

    # 多epsilon设置
    epsilons = [0 / 255, 4 / 255, 6 / 255, 8 / 255]
    step_sizes = [0, 1 / 255, 2 / 255, 2 / 255]  # 可适当调整步长
    repeat_num = 5

    all_results = {}

    for eps_idx, eps in enumerate(epsilons):
        print(f"\n========== Running for epsilon={eps} ==========\n")
        all_results[eps] = {'train_acc': [], 'test_acc': [], 'gap': []}

        for rep in range(repeat_num):
            print(f"\n---- Repeat {rep + 1}/{repeat_num} for epsilon={eps} ----\n")
            # 动态设定epsilon和步长到 args
            args.eps = eps
            args.sts = step_sizes[eps_idx]

            # 初始化数据集与用户
            train_dataset, test_dataset, user_groups = get_dataset(args)
            testloader = DataLoader(test_dataset, batch_size=args.local_bs, shuffle=False)

            # 构建模型
            if args.modeltype == 'NIN':
                global_model = NIN()
            elif args.modeltype == 'SmallCNN':
                global_model = SmallCNN()
            elif args.modeltype == 'resnet18':
                global_model = ResNet18()
            else:
                raise NotImplementedError

            global_model.to(device)
            global_weights = global_model.state_dict()

            if args.agg_opt == 'Scaffold':
                c_global_model = copy.deepcopy(global_model).cuda()
                nets = []
                for i in range(args.num_users):
                    net = copy.deepcopy(c_global_model)
                    nets.append(net)
            client_model = [copy.deepcopy(global_model).to(device) for i in range(args.num_users)]

            train_acc_list, test_acc_list, gap_list = [], [], []
            ipx = []
            for epoch in tqdm(range(args.epochs)):
                local_weights, local_losses, idt = [], [], []
                idx_train_acc = []
                ipp = []
                idx_num = []
                print(f'\n | Global Training Round : {epoch + 1} (eps={eps}, rep={rep + 1}) |\n')

                m = max(int(args.frac * args.num_users), 1)
                idxs_users = np.random.choice(range(args.num_users), m, replace=False)

                if args.agg_opt == 'Scaffold':
                    total_delta = copy.deepcopy(global_model.state_dict())
                    for key in total_delta:
                        total_delta[key] = 0.0

                for idx in idxs_users:
                    local_model = LocalUpdate(
                        args=args, dataset=train_dataset, idxs=user_groups[idx],
                        logger=logger, alg=args.agg_opt, anchor=global_model,
                        anchor_mu=args.mu, local_rank=ipx, method=args.train_method
                    )
                    client_model[idx] = copy.deepcopy(global_model)
                    if args.agg_opt == 'Scaffold':
                        w, loss, ide, idx_train, c_local_model, c_delta_para, pp_index = local_model.update_weights_scaffold(
                            copy.deepcopy(global_model), copy.deepcopy(c_global_model), nets[idx], global_round=epoch)
                        nets[idx] = copy.deepcopy(c_local_model)
                    else:
                        w, loss, ide, idx_train, pp_index = local_model.update_weights_at(
                            model=copy.deepcopy(client_model[idx]), global_round=epoch
                        )
                    local_weights.append(copy.deepcopy(w))
                    local_losses.append(copy.deepcopy(loss))
                    idt.append(ide)
                    ipp.append(pp_index)
                    idx_train_acc.append(idx_train)
                    if args.agg_opt == 'Scaffold':
                        for key in total_delta:
                            total_delta[key] += c_delta_para[key]
                ipx = idt
                if args.agg_opt == 'Scaffold':
                    for key in total_delta:
                        total_delta[key] /= len(idxs_users)
                    c_global_para = copy.deepcopy(c_global_model.state_dict())
                    for key in c_global_para:
                        if c_global_para[key].type() == 'torch.LongTensor':
                            c_global_para[key] += total_delta[key].type(torch.LongTensor)
                        elif c_global_para[key].type() == 'torch.cuda.LongTensor':
                            c_global_para[key] += total_delta[key].type(torch.cuda.LongTensor)
                        else:
                            c_global_para[key] += total_delta[key]
                    c_global_model.load_state_dict(c_global_para)

                # 聚合
                if args.agg_center == 'FedAvg':
                    global_weights = average_weights(local_weights)
                if args.agg_center == 'SFAT':
                    idt_sorted = np.sort(idt)
                    idtxnum = float('inf')
                    idtx = args.topk
                    if idtx > m:
                        idtx = m
                    if idtx != 0:
                        idtxnum = idt_sorted[m - idtx]
                    if epoch > 0:
                        global_weights = average_weights_alpha(local_weights, idt, idtxnum, args.pri)
                    else:
                        global_weights = average_weights(local_weights)
                global_model.load_state_dict(global_weights)

                # 全局平均训练精度
                list_acc, list_loss = [], []
                global_model.eval()
                for c in range(args.num_users):
                    local_model = LocalUpdate(args=args, dataset=train_dataset,
                                              idxs=user_groups[c], logger=logger, alg=args.agg_opt,
                                              anchor=global_model, anchor_mu=args.mu, local_rank=ipx)
                    acc, loss = local_model.inference(model=global_model)
                    list_acc.append(acc)
                    list_loss.append(loss)
                avg_train_acc = sum(list_acc) / len(list_acc)
                train_acc_list.append(avg_train_acc)

                # 测试集对抗精度
                if eps > 0:
                    # 对抗评估
                    _, test_adv_acc = attack.eval_robust(
                        global_model, testloader, perturb_steps=args.num_steps,
                        epsilon=eps, step_size=step_sizes[eps_idx], loss_fn="cent",
                        category="Madry", random=True)
                else:
                    # 普通评估
                    _, test_adv_acc = attack.eval_clean(global_model, testloader)
                test_acc_list.append(test_adv_acc)
                # 泛化gap
                gap = abs(test_adv_acc - avg_train_acc)
                gap_list.append(gap)
                print(
                    f'[Epoch {epoch + 1}] Train Acc={avg_train_acc:.4f} | Test Acc={test_adv_acc:.4f} | Gap={gap:.4f}')

            # 每次实验统计入总表
            all_results[eps]['train_acc'].append(train_acc_list)
            all_results[eps]['test_acc'].append(test_acc_list)
            all_results[eps]['gap'].append(gap_list)

            # 保存模型
            torch.save(global_model.state_dict(), os.path.join(args.out_dir, f'fedAT_epsilon_{eps}_rep_{rep + 1}.pth'))

            # ========== 新增：保存每个模型每次实验的pkl ==========
            single_result = {
                'train_acc': train_acc_list,
                'test_acc': test_acc_list,
                'gap': gap_list
            }
            with open(os.path.join(args.out_dir, f'fedAT_epsilon_{eps}_rep_{rep + 1}.pkl'), 'wb') as f:
                pickle.dump(single_result, f)

        # 每个epsilon单独保存
        with open(os.path.join(args.out_dir, f'results_epsilon_{eps}.pkl'), 'wb') as f:
            pickle.dump(all_results[eps], f)

    # 所有结果总表保存
    with open(os.path.join(args.out_dir, f'all_results_fedAT.pkl'), 'wb') as f:
        pickle.dump(all_results, f)

    print('\n所有训练完成，耗时 {:.2f} 秒'.format(time.time() - start_time))
