#!/usr/bin/env python
# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F

class PGDAttack:
    def __init__(self, model, epsilon=0.1, alpha=0.01, iters=10, 
                 restarts=1, norm='Linf', num_classes=2):
        self.model = model
        self.epsilon = epsilon
        self.alpha = alpha
        self.iters = iters
        self.restarts = restarts
        self.norm = norm
        self.num_classes = num_classes
        self.device = next(model.parameters()).device
        
    def perturb(self, x, y):
        """Generate adversarial examples using PGD attack"""
        x_adv = x.clone().detach().to(self.device)
        y = y.clone().detach().to(self.device)
        
        # For classification tasks
        loss_fn = nn.CrossEntropyLoss()
        
        # Multiple restarts
        for _ in range(self.restarts):
            # Random initialization within epsilon ball
            if self.norm == 'Linf':
                x_adv = x_adv + torch.zeros_like(x_adv).uniform_(-self.epsilon, self.epsilon)
            else:
                raise NotImplementedError("Only Linf norm implemented")
            
            x_adv = torch.clamp(x_adv, 0, 1)  # assuming input is in [0,1]
            
            for _ in range(self.iters):
                x_adv.requires_grad_(True)
                outputs = self.model(x_adv)
                
                # Calculate loss
                loss = loss_fn(outputs, y)
                
                # Compute gradient
                grad = torch.autograd.grad(loss, x_adv)[0]
                
                # Update adversarial examples
                if self.norm == 'Linf':
                    x_adv = x_adv.detach() + self.alpha * torch.sign(grad.detach())
                    # Project back to epsilon ball
                    x_adv = torch.min(torch.max(x_adv, x - self.epsilon), x + self.epsilon)
                    # Clip to valid pixel range
                    x_adv = torch.clamp(x_adv, 0, 1)
                else:
                    raise NotImplementedError("Only Linf norm implemented")
        
        return x_adv