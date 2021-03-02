import torch
import torch.nn as nn
from typing import Any


__all__ = ['alexnet']



class AlexNet(nn.Module):

    def __init__(self, num_classes: int = 1000) -> None:
        super(AlexNet, self).__init__()
        # TODO: implement alex net for task 1; 
        # You are free to change channels, kernel sizes, strides, etc. But the model's flops must be smaller than 200M.

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x


def alexnet(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> AlexNet:
    model = AlexNet(**kwargs)
    return model
