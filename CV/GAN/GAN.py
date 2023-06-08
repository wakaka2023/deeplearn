from torch import nn


class Generator(nn.Module):
    def __init__(self, img_sz):
        self.img_size = img_sz
        super(Generator, self).__init__()
        self.gener = nn.Sequential(
            nn.Linear(100, 256),
            nn.ReLU(True),
            nn.Linear(256, 512),
            nn.ReLU(True),
            nn.Linear(512, img_sz * img_sz * 1),
            nn.Tanh(),
        )

    def forward(self, inputs):
        outputs = self.gener(inputs)
        outputs = outputs.view(-1, self.img_size, self.img_size, 1)
        return outputs


class Discriminator(nn.Module):
    def __init__(self, img_sz):
        self.img_size = img_sz
        super(Discriminator, self).__init__()
        self.discri = nn.Sequential(
            nn.Linear(img_sz * img_sz, 512),
            nn.LeakyReLU(True),
            nn.Linear(512, 256),
            nn.LeakyReLU(True),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, inputs):
        outputs = inputs.view(-1, self.img_size * self.img_size)
        outputs = self.discri(outputs)
        return outputs
