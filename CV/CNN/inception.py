import torch
from torch import nn


class BasicConv(nn.Module):
    def __init__(self, in_channel, out_channel, **kwargs):
        super().__init__()
        self.conv = nn.Conv2d(in_channel, out_channel, bias=True)
        self.bn = nn.BatchNorm2d(out_channel)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = torch.relu(x)

        return x


class InceptionBasicBlock(nn.Module):
    def __init__(self, in_channel, pool_features):
        super().__init__()
        self.branch1x1 = BasicConv(in_channel, 64, kernel_size=1)

        self.branch5x5_1 = BasicConv(in_channel, 16, kernel_size=1)
        self.branch5x5_2 = BasicConv(16, 32, kernel_size=5, padding=2)

        self.branch3x3_1 = BasicConv(in_channel, 96, kernel_size=1)
        self.branch3x3_2 = BasicConv(96, 128, kernel_size=3)

        self.branch1x1_pool = BasicConv(pool_features, 32, kernel_size=1)

    def forward(self, x):
        branch1x1 = self.branch1x1(x)

        branch3x3 = self.branch3x3_1(x)
        branch3x3 = self.branch3x3_2(branch3x3)

        branch5x5 = self.branch5x5_1(x)
        branch5x5 = self.branch5x5_2(branch5x5)

        branch_pool = nn.AvgPool2d(x, kernel_size=3, stride=1, padding=1)
        branch_pool = self.branch1x1_pool(branch_pool)

        return branch1x1 + branch3x3 + branch5x5 + branch_pool
