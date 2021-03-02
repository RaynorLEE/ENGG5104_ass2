import torch
import torch.nn as nn
import numpy as np

class CrossEntropyLoss(nn.Module):
      def __init__(self, **kwargs):
          super(CrossEntropyLoss, self).__init__()
          # TODO: implemente cross entropy loss for task2;
          # You cannot directly use any loss functions from torch.nn or torch.nn.functional, other modules are free to use.

      def forward(x, y, **kwargs):
          loss = 0
          return loss


