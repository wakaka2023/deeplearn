import copy
import os.path
from datetime import datetime

import torch
from tqdm import tqdm

from CV.template.utils.model import get_model, load_ckpt


def fit(epochs, model, train_dl, test_dl, lr, save_path, device_name, resume, num_class):
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    device = torch.device(device_name)
    print(f'Use: {device}')

    model, loss_fn, optimizer, scheduler = get_model(lr, model, num_class)

    start, best_acc = 0, .0
    Train_Acc, Train_Loss = [], []
    Test_Acc, Test_Loss = [], []
    if resume:
        start, best_acc, model, optimizer, scheduler, epoch_data = load_ckpt(save_path, device_name, model, optimizer,
                                                                             scheduler)
        Train_Loss, Train_Acc, Test_Loss, Test_Acc = epoch_data.values()

    model.to(device)
    for i in range(start, epochs):
        print(f"--------------epoch [{i + 1}/{epochs}] --------------{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

        model.train()
        epoch_acc, epoch_loss = 0, 0
        batches, sizes = len(train_dl), len(train_dl.dataset)
        for imgs, labels in tqdm(train_dl):
            imgs, labels = imgs.to(device), labels.to(device)
            predict_label = model(imgs)
            try:
                loss = loss_fn(predict_label, labels)
            except:
                # inception_v3
                predict_label = predict_label.logits
                loss = loss_fn(predict_label, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            with torch.no_grad():
                epoch_loss += loss.item()
                epoch_acc += (predict_label.argmax(1) == labels).sum().item()

        Train_Acc.append(epoch_acc / sizes)
        Train_Loss.append(epoch_loss / batches)

        model.eval()
        epoch_acc, epoch_loss = 0, 0
        batches, sizes = len(test_dl), len(test_dl.dataset)
        with torch.no_grad():
            for imgs, labels in tqdm(test_dl):
                imgs, labels = imgs.to(device), labels.to(device)
                predict_label = model(imgs)
                loss = loss_fn(predict_label, labels)

                epoch_loss += loss.item()
                epoch_acc += (predict_label.argmax(1) == labels).sum().item()

        Test_Acc.append(epoch_acc / sizes)
        Test_Loss.append(epoch_loss / batches)

        print("Train:  Loss: {:.4f}  Acc: {:.2f}%  lr: {}".format(Train_Loss[i], Train_Acc[i] * 100,
                                                                  optimizer.param_groups[0]['lr']))
        print("Valid:  Loss: {:.4f}  Acc: {:.2f}%".format(Test_Loss[i], Test_Acc[i] * 100))

        if Test_Acc[i] > best_acc:
            best_model_wts = copy.deepcopy(model.state_dict())
            best_acc = Test_Acc[i]
            torch.save(best_model_wts, f'{save_path}/best.pt')
            print("reach best acc: {:.2f}%".format(Test_Acc[i] * 100))

        torch.save(model.state_dict(), f'{save_path}/last.pt')

        epoch_data = {'train_loss': Train_Loss,
                      'train_acc': Train_Acc,
                      'test_loss': Test_Loss,
                      'test_acc': Test_Acc
                      }
        torch.save({
            'epoch': i + 1,
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict() if scheduler is not None else None,
            'epoch_data': epoch_data,
            'best_acc': best_acc,
        }, f'{save_path}/epoch_data.pt')

        if scheduler is not None:
            scheduler.step()

    print('Train Complete!')
