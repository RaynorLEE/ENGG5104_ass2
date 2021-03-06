import torch
import torch.nn as nn
import numpy as np


class CrossEntropyLoss(nn.Module):
    def __init__(self, **kwargs):
        super(CrossEntropyLoss, self).__init__()
        # TODO: implemente cross entropy loss for task2;
        # You cannot directly use any loss functions from torch.nn or torch.nn.functional, other modules are free to use.
        self.task = 2
        if 'cls_count' in kwargs:
            self.task = 4
            cls_count = kwargs['cls_count']
            N = kwargs['dataset_size']
            beta = float(N - 1) / N
            device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
            self.weight = torch.zeros(size=[len(cls_count)], dtype=torch.float32, device=device)
            for i in range(len(cls_count)):
                n_y = cls_count[i]
                weight = (1. - beta) / (1. - np.power(beta, n_y))
                self.weight[i] = weight
            self.weight.requires_grad = True
            #   For debugging
            print('loss weight = [', end='')
            for i in range(len(cls_count)):
                print(self.weight[i], end=' ')
            print(']')

    def forward(self, x, y, epsilon=1e-12, **kwargs):
        #   x = torch.einsum('c,bc->bc', self.weight, x)
        if self.task == 4:
            x = self.weight * x
        log_sum_exp = torch.logsumexp(x, dim=1)
        y = torch.unsqueeze(y, 1)
        x = torch.gather(dim=1, input=x, index=y)
        loss = -x + log_sum_exp
        return loss.mean()
