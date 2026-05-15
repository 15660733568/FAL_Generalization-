#此代码参考''Uniformly Stable Algorithms for Adversarial Training and Beyond''论文代码。如要复现，学习相关内容


import torch
from torch import nn
import copy
from models import PGD_Linf_Attack

class LocalUpdateFalME(object):
    def __init__(self, args, dataset, idxs, epsilon):
        self.args = args
        self.trainloader = torch.utils.data.DataLoader(
            DatasetSplit(dataset, idxs), batch_size=args.local_bs, shuffle=True
        )
        self.device = 'cuda' if args.gpu else 'cpu'
        self.criterion = nn.NLLLoss().to(self.device)
        self.epsilon = epsilon
        self.rho = args.me_rho      # Moreau 强度 rho
        self.alpha = args.me_alpha  # 步长 alpha

    def update_weights(self, model_w, model_u):
        """
        对应伪代码 Step 7-14
        model_w: 当前全局广播的工作模型 w^t
        model_u: 该客户端持有的本地 Moreau 参数 u_i^t
        """
        model_w.train()
        model_u.to(self.device)
        optimizer = torch.optim.SGD(model_w.parameters(), lr=self.args.lr, momentum=self.args.momentum)
        
        # Step 9: 初始化攻击器
        pgd_attack = PGD_Linf_Attack(model_w, epsilon=self.epsilon, steps=self.args.pgd_steps)

        for _ in range(self.args.local_ep):
            for images, labels in self.trainloader:
                images, labels = images.to(self.device), labels.to(self.device)
                
                # Step 9: 生成对抗样本 z_adv
                adv_images = pgd_attack.perturb(images, labels)
                
                model_w.zero_grad()
                outputs = model_w(adv_images)
                
                # Step 10: 计算损失。根据 ME-A 理论，需包含近端项 (Proximal Term)
                # Loss = Adversarial_Loss + (rho/2) * ||w - u||^2
                adv_loss = self.criterion(outputs, labels)
                prox_loss = (self.rho / 2.0) * self.calculate_prox_term(model_w, model_u)
                total_loss = adv_loss + prox_loss
                
                total_loss.backward()
                optimizer.step()

        # Step 14: 更新本地 Moreau 参数 u_i^{t+1} = u_i^t + alpha * (w_i^{t+1} - u_i^t)
        self.update_u(model_u, model_w)

        # Step 15: 返回更新后的 w 和 u 到服务端
        return model_w.state_dict(), model_u.state_dict(), total_loss.item()

    def calculate_prox_term(self, w, u):
        prox = 0.0
        for p_w, p_u in zip(w.parameters(), u.parameters()):
            prox += torch.sum((p_w - p_u) ** 2)
        return prox

    def update_u(self, u, w):
        with torch.no_grad():
            for p_u, p_w in zip(u.parameters(), w.parameters()):
                p_u.data += self.alpha * (p_w.data - p_u.data)

class DatasetSplit(torch.utils.data.Dataset):
    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = [int(i) for i in idxs]
    def __len__(self): return len(self.idxs)
    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        return image.clone(), torch.tensor(label)
