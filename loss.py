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
        log_sum_exp = torch.logsumexp(x, dim=1)
        y = torch.unsqueeze(y, 1)
        x = torch.gather(dim=1, input=x, index=y)
        loss = -x + log_sum_exp
        return loss.mean()
