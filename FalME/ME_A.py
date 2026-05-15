#此代码参考''Uniformly Stable Algorithms for Adversarial Training and Beyond''论文代码，如要复现，请学习相关内容


import torch.nn as nn
import copy

def prox_term(model, center_model):
    """
    计算近端项 (Proximal Term): (rho/2) * ||w - u||^2
    用于内循环优化工作模型 w。
    """
    prox = 0.0
    # 遍历所有参数，计算欧几里得距离的平方
    for param_w, param_u in zip(model.parameters(), center_model.parameters()):
        prox += torch.sum((param_w - param_u) ** 2)
    return prox

def step_u(center_model, model, alpha):
    """
    外循环更新影子模型 u: u^{t+1} = u^t + alpha * (w^{t+1} - u^t)
    alpha 对应伪代码中的步长。
    """
    with torch.no_grad():
        for param_u, param_w in zip(center_model.parameters(), model.parameters()):
            # 执行移动平均更新
            param_u.data += alpha * (param_w.data - param_u.data)







def train_mea(args, model, train_loader):
    # 1. 初始化: 工作模型 w 和影子模型 u
    w = model
    u = copy.deepcopy(model)
    u.requires_grad_(False) # u 不参与自动求导计算
    
    optimizer = torch.optim.SGD(w.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()
    rho = args.me_rho      # 近端项强度系数
    alpha = args.me_alpha  # 外循环更新步长

    # 3. 外循环 (Communication/Outer Rounds)
    for t in range(args.epochs):
        w.train()
        
        # b. 内循环 (Local Steps / Inner Epochs)
        for images, labels in train_loader:
            optimizer.zero_grad()
            
            # 计算任务损失 (Task Loss)
            outputs = w(images)
            task_loss = criterion(outputs, labels)
            
            # 计算近端损失 (Proximal Loss)
            # 这里的 prox_term 就是算法的核心约束
            p_loss = (rho / 2.0) * prox_term(w, u)
            
            # 总损失: K(w, u; S)
            total_loss = task_loss + p_loss
            
            total_loss.backward()
            optimizer.step()
        
        # d. 更新中心模型 u (Step U)
        # 每一个外循环结束，将 w 的轨迹同步给 u
        step_u(u, w, alpha)
        
        print(f"Round {t} completed. Loss: {total_loss.item():.4f}")

    return w, u
