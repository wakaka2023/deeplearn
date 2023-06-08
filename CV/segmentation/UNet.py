import torch
from torch import nn


class DownSample(nn.Module):
    def __init__(self, c1, c2):
        super(DownSample, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(c1, c2, 3, padding=1),
            nn.BatchNorm2d(c2),
            nn.ReLU(True),

            nn.Conv2d(c2, c2, 3, padding=1),
            nn.BatchNorm2d(c2),
            nn.ReLU(True)
        )
        self.pool = nn.MaxPool2d(2)

    def forward(self, inputs, pool=True):
        if pool:
            inputs = self.pool(inputs)
        output = self.conv(inputs)
        return output


class UpSample(nn.Module):
    def __init__(self, c1, c2, up_channel=0):
        """
        When catLayer is only one(not list),c1 equals up_channel,and is ConvTranspose2d input_channel
        otherwise(is list,means concat multiple layer) up_channel is ConvTranspose2d input_channel,that is not equals c1

        :param c1:Input channel
        :param c2:Output channel
        :param up_channel:ConvTranspose2d input_channel
        """
        super(UpSample, self).__init__()
        if up_channel:  # ip_channel != 0
            self.up_conv = nn.Sequential(
                nn.ConvTranspose2d(up_channel, c2, 3, stride=2, padding=1, output_padding=1),
                nn.ReLU(True)
            )
        else:  # ip_channel == 0
            self.up_conv = nn.Sequential(
                nn.ConvTranspose2d(c1, c2, 3, stride=2, padding=1, output_padding=1),
                nn.ReLU(True)
            )
        self.conv = nn.Sequential(
            nn.Conv2d(c1, c2, 3, padding=1),
            nn.BatchNorm2d(c2),
            nn.ReLU(True),
            nn.Conv2d(c2, c2, 3, padding=1),
            nn.BatchNorm2d(c2),
            nn.ReLU(True)
        )

    def forward(self, inputs, cat_layer):
        out_put = self.up_conv(inputs)

        out_put = torch.cat(cat_layer + [out_put], dim=1) if isinstance(cat_layer, list) else torch.cat(
            [out_put, cat_layer], dim=1)
        out_put = self.conv(out_put)
        return out_put


class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()
        self.d1 = DownSample(3, 64)
        self.d2 = DownSample(64, 128)
        self.d3 = DownSample(128, 256)
        self.d4 = DownSample(256, 512)
        self.d5 = DownSample(512, 1024)
        self.u4 = UpSample(1024, 512)
        self.u3 = UpSample(512, 256)
        self.u2 = UpSample(256, 128)
        self.u1 = UpSample(128, 64)
        self.conv1_1 = nn.Conv2d(64, 3, 1)

    def forward(self, inputs):
        d1 = self.d1(inputs, pool=False)
        d2 = self.d2(d1)
        d3 = self.d3(d2)
        d4 = self.d4(d3)
        conv = self.d5(d4)
        u4 = self.u4(conv, d4)
        u3 = self.u3(u4, d3)
        u2 = self.u2(u3, d2)
        u1 = self.u1(u2, d1)
        output = self.conv1_1(u1)
        return output


'''
d1                u1 --> output
  d2            u2
    d3        u3
      d4    u4
        conv
'''
