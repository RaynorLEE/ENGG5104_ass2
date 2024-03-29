import torch
import torch.nn as nn
from typing import Any


__all__ = ['alexnet']


class AlexNet(nn.Module):

    def __init__(self, num_classes: int = 1000) -> None:
        super(AlexNet, self).__init__()
        # TODO: implement alex net for task 1; 
        # You are free to change channels, kernel sizes, strides, etc. But the model's flops must be smaller than 200M.
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1),
            nn.ReLU()
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1),
            nn.ReLU()
        )
        self.conv5 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2)
        )
        self.avg_pool = nn.AdaptiveAvgPool2d((2, 2))
        self.fc1 = nn.Sequential(
            nn.Linear(1024, 2048),
            nn.ReLU()
        )
        self.fc2 = nn.Sequential(
            nn.Linear(2048, 1024),
            nn.ReLU()
        )
        self.fc3 = nn.Linear(1024, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.avg_pool(x)
        x = x.view(-1, 1024)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x


def alexnet(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> AlexNet:
    model = AlexNet(**kwargs)
    return model
