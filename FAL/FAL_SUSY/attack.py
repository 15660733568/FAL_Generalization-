import torch

def pgd_linf(model, images, labels, epsilon, alpha, iters, random_start=True, clip_min=0.0, clip_max=1.0):
    """
    Projected Gradient Descent (L-infinity) adversarial attack

    Args:
        model: 要攻击的模型
        images: 原始图像 [B,C,H,W], 必须已归一化（torch tensor）
        labels: 正确标签 [B]
        epsilon: 扰动范围 (float, e.g., 8/255)
        alpha: 单步步长 (float, e.g., 2/255)
        iters: 迭代步数 (int)
        random_start: 是否加初始扰动
        clip_min: 最小像素值 (float)
        clip_max: 最大像素值 (float)
    Returns:
        对抗样本（tensor）
    """

    model.eval()
    images = images.clone().detach().to(next(model.parameters()).device)
    labels = labels.clone().detach().to(images.device)
    ori_images = images.data

    if random_start:
        images = images + torch.empty_like(images).uniform_(-epsilon, epsilon)
        images = torch.clamp(images, min=clip_min, max=clip_max)

    for i in range(iters):
        images.requires_grad = True
        outputs = model(images)
        loss = torch.nn.functional.cross_entropy(outputs, labels)
        grad = torch.autograd.grad(loss, images, retain_graph=False, create_graph=False)[0]
        adv_images = images + alpha * grad.sign()
        eta = torch.clamp(adv_images - ori_images, min=-epsilon, max=epsilon)
        images = torch.clamp(ori_images + eta, min=clip_min, max=clip_max).detach()

    return images
