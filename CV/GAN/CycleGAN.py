import torch
from torch import nn
import torch.nn.functional as F


class CIL(nn.Module):
    def __init__(self, c1, c2, **kwargs):
        super(CIL, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(c1, c2, **kwargs),
            nn.InstanceNorm2d(c2),
            nn.LeakyReLU(0.2, inplace=True)
        )

    def forward(self, x):
        return self.main(x)


class CTIR(nn.Module):
    def __init__(self, c1, c2):
        super(CTIR, self).__init__()
        self.main = nn.Sequential(
            nn.ConvTranspose2d(c1, c2, 3, 2, 1, 1),
            nn.InstanceNorm2d(c2),
            nn.ReLU(True)
        )

    def forward(self, x):
        return self.main(x)


class ResidualBlock(nn.Module):
    def __init__(self, c):
        super(ResidualBlock, self).__init__()
        self.residual = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(c, c, 3),
            nn.InstanceNorm2d(c),
            nn.ReLU(True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(c, c, 3),
            nn.InstanceNorm2d(c),
        )

    def forward(self, x):
        return x + self.residual(x)


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.main = nn.Sequential(
            nn.ReflectionPad2d(3),
            CIL(3, 64, kernel_size=7),
            CIL(64, 128, kernel_size=3, stride=2, padding=1),
            CIL(128, 256, kernel_size=3, stride=2, padding=1),
            *[ResidualBlock(256)] * 9,
            CTIR(256, 128),
            CTIR(128, 64),
            nn.ReflectionPad2d(3),
            nn.Conv2d(64, 3, 7, 1),
            nn.Tanh()
        )

    def forward(self, x):
        return self.main(x)


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(3, 64, 4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),

            CIL(64, 128, kernel_size=4, stride=2, padding=1),
            CIL(128, 256, kernel_size=4, stride=2, padding=1),
            CIL(256, 512, kernel_size=4, stride=2, padding=1),
            nn.Conv2d(512, 1, 4, 1, 1)
        )

    def forward(self, x):
        x = self.main(x)
        return F.avg_pool2d(x, x.size()[2:]).view(x.size()[0], -1)
        # return x


def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm2d') != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)


if __name__ == '__main__':
    t = torch.zeros((1, 3, 64, 64))
    # model = Generator()
    model = Discriminator()
    print(model(t).shape)
    # print(model.parameters)
