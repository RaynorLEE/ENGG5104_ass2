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
            #   N = kwargs['dataset_size']
            #   beta = float(N - 1) / N
            beta = 0.9999
            device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
            # self.weight = torch.ones(size=[len(cls_count)], dtype=torch.float32, device=device)
            weight = []
            weight_sum = 0.0
            for i in range(len(cls_count)):
                n_y = cls_count[i]
                w = (1. - beta) / (1. - np.power(beta, n_y))
                weight.append(w)
                weight_sum += w
            for i in range(len(cls_count)):
                weight[i] = (weight[i] / weight_sum) * 10
            #   For debugging ONLY
            print('loss weight = [', end='')
            for i in range(len(cls_count)):
                print(weight[i], end=' ')
            print(']')

            self.weight = torch.as_tensor(data=weight, dtype=torch.float32, device=device)


    def forward(self, x, y, epsilon=1e-12, **kwargs):
        #   x = torch.einsum('c,bc->bc', self.weight, x)
        # if self.task == 4:
        #     x = self.weight * x
        #     #   x = torch.einsum('c,bc->bc', self.weight, x)
        log_sum_exp = torch.logsumexp(x, dim=1)
        y = torch.unsqueeze(y, 1)
        x = torch.gather(dim=1, input=x, index=y)
        x = x.squeeze(-1)
        loss = -x + log_sum_exp
        if self.task == 4:
            curr_weight = torch.zeros(loss.size(), dtype=x.dtype, device=x.device)
            for i in range(len(y)):
                curr_weight[i] = self.weight[y[i]]
            loss = curr_weight * loss
        #     result = loss.sum() / curr_weight.sum()
        # else:
        #     result = loss.mean()
        # return result
        return loss.mean()
