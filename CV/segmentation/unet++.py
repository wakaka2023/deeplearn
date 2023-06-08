import numpy as np
import torch
from torch import nn
from UNet import DownSample, UpSample


class UnetPlus(nn.Module):
    def __init__(self, num_class=3):
        super(UnetPlus, self).__init__()
        self.d0_0 = DownSample(3, 64)
        self.d1_0 = DownSample(64, 128)
        self.d2_0 = DownSample(128, 256)
        self.d3_0 = DownSample(256, 512)
        self.d4_0 = DownSample(512, 1024)

        self.u0_1 = UpSample(64 * 2, 64)
        self.u0_2 = UpSample(64 * 3, 64, 128)
        self.u0_3 = UpSample(64 * 4, 64, 128)
        self.u0_4 = UpSample(64 * 5, 64, 128)

        self.u1_1 = UpSample(128 * 2, 128)
        self.u1_2 = UpSample(128 * 3, 128, 256)
        self.u1_3 = UpSample(128 * 4, 128, 256)

        self.u2_1 = UpSample(256 * 2, 256)
        self.u2_2 = UpSample(256 * 3, 256, 512)

        self.u3_1 = UpSample(512 * 2, 512)

        self.conv1_1 = nn.Conv2d(64, num_class, 1)

    def forward(self, inputs):
        x0_0 = self.d0_0(inputs, pool=False)
        x1_0 = self.d1_0(x0_0)
        x2_0 = self.d2_0(x1_0)
        x3_0 = self.d3_0(x2_0)
        x4_0 = self.d4_0(x3_0)

        x0_1 = self.u0_1(x1_0, x0_0)
        x1_1 = self.u1_1(x2_0, x1_0)
        x2_1 = self.u2_1(x3_0, x2_0)
        x3_1 = self.u3_1(x4_0, x3_0)

        x0_2 = self.u0_2(x1_1, [x0_1, x0_0])
        x1_2 = self.u1_2(x2_1, [x1_1, x1_0])
        x2_2 = self.u2_2(x3_1, [x2_1, x2_0])

        x0_3 = self.u0_3(x1_2, [x0_2, x0_1, x0_0])
        x1_3 = self.u1_3(x2_2, [x1_2, x1_1, x1_0])

        x0_4 = self.u0_4(x1_3, [x0_3, x0_2, x0_1, x0_0])

        output1 = self.conv1_1(x0_1)
        output2 = self.conv1_1(x0_2)
        output3 = self.conv1_1(x0_3)
        output4 = self.conv1_1(x0_4)

        return [output1, output2, output3, output4]


if __name__ == '__main__':
    img = torch.tensor(np.zeros((1, 3, 256, 256))).float()
    model = UnetPlus()
    print(model(img))
