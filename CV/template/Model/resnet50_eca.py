import torch
import torchvision
from torch import nn

from CV.template.Model.ECANet import eca_layer


# from utils.ECANet import eca_layer


def built_model():
    model, loss_fn, optimizer, scheduler = None, None, None, None
    """
    Build the Model and others here
    """
    model = torchvision.models.resnet50(pretrained=True)
    model.fc = nn.Linear(2048, 24, bias=True)

    for i in range(1, 5):
        layer = eval(f'Model.layer{i}')
        for j in range(len(layer)):
            block = layer[j]
            c = block.conv3.out_channels
            conv = block.conv3
            block.conv3 = nn.Sequential(
                conv,
                eca_layer(c)
            )
    for i in range(1, 5):
        layer = eval(f'Model.layer{i}')
        for j in range(len(layer)):
            block = layer[j]
            c = block.downsample[0].out_channels
            conv = block.downsample[0]
            block.downsample[0] = nn.Sequential(
                conv,
                eca_layer(c)
            )
            break

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=40, gamma=0.1)

    return model, loss_fn, optimizer, scheduler