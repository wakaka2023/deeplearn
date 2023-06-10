import torch
import torchvision
from torch import nn


class TrainModel:
    def __init__(self, num_class, lr):
        self.num_class = num_class
        self.lr = lr

    def model(self):
        model = torchvision.models.resnet18(pretrained=True)

        for param in model.parameters():
            param.requires_grad = False
        for param in model.fc.parameters():
            param.requires_grad = True
        model.fc = nn.Linear(512, self.num_class, bias=True)
        return model

    def loss_fn(self):
        return nn.CrossEntropyLoss()

    def optimizer(self):
        model = self.model()
        optimizer = torch.optim.Adam(model.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=40, gamma=0.1)
        return optimizer, scheduler

    def built_model(self):
        model = self.model()
        loss_fn = self.loss_fn()
        optimizer, scheduler = self.optimizer()

        return model, loss_fn, optimizer, scheduler
