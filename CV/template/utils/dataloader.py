import torch
import torchvision
from torch.utils.data import DataLoader
from torchvision import transforms


def dataloader(opt, trans):
    is_cuda = True if opt.device != 'cpu' else False
    train_ds = torchvision.datasets.ImageFolder(
        opt.traindata,
        transform=trans
    )
    test_ds = torchvision.datasets.ImageFolder(
        opt.testdata,
        transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((opt.imgsz, opt.imgsz))
        ])
    )
    train_dl = DataLoader(
        train_ds,
        batch_size=opt.batch,
        shuffle=True,
        num_workers=opt.workers if is_cuda else 0,
        pin_memory=is_cuda
    )
    test_dl = torch.utils.data.DataLoader(
        test_ds,
        batch_size=opt.batch,
        num_workers=opt.workers if is_cuda else 0,
        pin_memory=is_cuda
    )
    print(
        f'Train Size: {len(train_ds)}  Test Size: {len(test_ds)}  Batch Size: {opt.batch}  Num Class: {len(train_ds.classes)}')
    return train_dl, test_dl, len(train_ds.classes)
