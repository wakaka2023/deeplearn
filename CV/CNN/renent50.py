import torch
from torch import nn


class Bottleneck(nn.Module):
    def __init__(self, c1, c2, c3, s=1, isdown=False):
        super(Bottleneck, self).__init__()
        self.residual = nn.Sequential(
            nn.Conv2d(c1, c2, 1, bias=False),
            nn.BatchNorm2d(c2),
            nn.ReLU(True),
            nn.Conv2d(c2, c2, 3, stride=s, padding=1, bias=False),
            nn.BatchNorm2d(c2),
            nn.ReLU(True),
            nn.Conv2d(c2, c3, 1, bias=False),
            nn.BatchNorm2d(c3),
            nn.ReLU(True)
        )
        self.downsample = nn.Sequential(
            nn.Conv2d(c1, c3, 1, stride=s, bias=False),
            nn.BatchNorm2d(c3)
        )
        self.isdown = isdown

    def forward(self, x):
        x_shortcut = self.downsample(x) if self.isdown else x
        print('')
        return x_shortcut + self.residual(x)


class ResNet50(nn.Module):
    def __init__(self, num_class=1000):
        super(ResNet50, self).__init__()
        self.layer0 = nn.Sequential(
            nn.Conv2d(3, 64, 7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.MaxPool2d(3, stride=2, padding=1)
        )
        self.layer1 = nn.Sequential(
            Bottleneck(64, 64, 256, isdown=True),
            *[Bottleneck(256, 64, 256)] * 2,
        )
        self.layer2 = nn.Sequential(
            Bottleneck(256, 128, 512, s=2, isdown=True),
            *[Bottleneck(512, 128, 512)] * 3,
        )
        self.layer3 = nn.Sequential(
            Bottleneck(512, 256, 1024, s=2, isdown=True),
            *[Bottleneck(1024, 256, 1024)] * 5,
        )
        self.layer4 = nn.Sequential(
            Bottleneck(1024, 512, 2048, s=2, isdown=True),
            *[Bottleneck(2048, 512, 2048)] * 2,
        )
        self.layer5 = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.fc = nn.Linear(2048, num_class)

    def forward(self, x):
        outputs = self.layer0(x)
        outputs = self.layer1(outputs)
        outputs = self.layer2(outputs)
        outputs = self.layer3(outputs)
        outputs = self.layer4(outputs)
        outputs = self.layer5(outputs).squeeze(dim=[-1, -2])
        outputs = self.fc(outputs)
        return outputs


if __name__ == '__main__':
    model = ResNet50()
    # x = torch.zeros(1, 3, 224, 224)
    # print(model(x).shape)
    print(model)
