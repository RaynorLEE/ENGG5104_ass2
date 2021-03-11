import torch.nn as nn
import torch.nn.functional as F


class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )
        self.bypass_path1 = nn.Sequential()
        if in_channels < out_channels:
            self.bypass_path1 = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size,
                          stride=stride, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU()
            )
        self.conv3 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )
        self.bypass_path2 = nn.Sequential()

    def forward(self, x):
        res_x = x
        x = self.conv1(x)
        x = self.conv2(x)
        x += self.bypass_path1(res_x)
        res_x = x
        x = self.conv3(x)
        x = self.conv4(x)
        x += self.bypass_path2(res_x)
        return x


class ResNet(nn.Module):
    def __init__(self):
        super(ResNet, self).__init__()
        #   input image: 3 x 32 x 32
        #   data pre-processing
        self.init = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(32),
            nn.ReLU()
        )
        #   Residual network for each stage
        self.stage1 = ResBlock(in_channels=32, out_channels=32, kernel_size=3, stride=(1,))
        self.stage2 = ResBlock(in_channels=32, out_channels=64, kernel_size=3, stride=(2,))
        self.stage3 = ResBlock(in_channels=64, out_channels=128, kernel_size=3, stride=(2,))
        self.stage4 = ResBlock(in_channels=128, out_channels=256, kernel_size=3, stride=(2,))
        self.pool = nn.AvgPool2d(4, 4)
        #   Full Connection Layers
        self.fc = nn.Linear(256, 10)

    def forward(self, x):
        x = self.init(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.pool(x)
        x = x.view(x.size(0), 256 * 1 * 1)
        x = self.fc(x)
        return x
