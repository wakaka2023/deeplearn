import argparse

from utils.dataloader import dataloader
from torchvision import transforms

from warnings import filterwarnings
from utils.train import fit

filterwarnings('ignore')


def main(opt):
    transform = transforms.Compose([
        transforms.Resize((opt.imgsz, opt.imgsz)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomRotation(degrees=(-20, 70)),
        # transforms.ColorJitter(
        #     brightness=(0.7, 1.3),
        #     contrast=(0.7, 1.3),
        #     saturation=(0.7, 1.3),
        #     hue=(-0.05, 0.05)),
        transforms.ToTensor(),
        # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    train_dl, test_dl,num_class = dataloader(opt, trans=transform)
    fit(
        epochs=opt.epochs,
        train_dl=train_dl,
        test_dl=test_dl,
        save_path=opt.savepath,
        device_name=opt.device,
        lr=opt.lr,
        resume=opt.resume,
        model=opt.model,
        num_class = num_class
    )


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--savepath', type=str, default='logs',
                        help='The directory path where the checkpoint is saved')
    parser.add_argument('--epochs', type=int, default=100, help='Total epochs of training')
    parser.add_argument('--batch', type=int, default=8, help='Batch size')
    parser.add_argument('--workers', type=int, default=8, help='Num workers')
    parser.add_argument('--resume', nargs='?', const=True, default=False, help='Resume training')
    parser.add_argument('--lr', type=float, default=0.001, help='Learn rate')
    parser.add_argument('--traindata', type=str, required=True, help='Training set directory')
    parser.add_argument('--testdata', type=str, required=True, help='Test set directory')
    parser.add_argument('--imgsz', type=int, default=256, help='Image size')
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--model', type=str, default='resnet18')
    return parser.parse_args()


if __name__ == '__main__':
    opt = parse_opt()
    main(opt)
