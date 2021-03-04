import torch
import torch.nn as nn
import numpy as np


class CrossEntropyLoss(nn.Module):
    def __init__(self, **kwargs):
        super(CrossEntropyLoss, self).__init__()
        # TODO: implemente cross entropy loss for task2;
        # You cannot directly use any loss functions from torch.nn or torch.nn.functional, other modules are free to use.
        #   self.loss_func = nn.CrossEntropyLoss()

    def forward(self, x, y, epsilon=1e-12, **kwargs):
        # x = torch.clamp(x, 0, 1.0-epsilon)
        # exp_x = torch.exp(x)
        # mat = torch.zeros(exp_x.size(), device=exp_x.device)
        # for i in range(y.size()[0]):
        #     mat[i][y[i]] = 1
        # loss = -torch.sum(torch.mul(mat, x), dim=-1) + torch.log(torch.sum(exp_x, dim=-1))
        x = torch.clamp(x, 0, 1.0 - epsilon)
        exp_sum = torch.sum(torch.exp(x), dim=-1)
        y = torch.unsqueeze(y, 1)
        x = torch.gather(dim=1, input=x, index=y)
        loss = -x + torch.log(exp_sum)
        return loss.mean()
        #   return self.loss_func(x, y)
