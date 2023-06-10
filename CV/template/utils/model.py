import torch
import importlib


def get_model(lr, model, num_class):
    module = importlib.import_module(f"CV.template.Model.{model}")
    obj = module.TrainModel(lr=lr, num_class=num_class)
    return obj.built_model()


def load_ckpt(save_path, device_name, model, optimizer, scheduler=None):
    last_model = torch.load(f'{save_path}/last.pt', map_location=device_name)
    ckpt = torch.load(f'{save_path}/epoch_data.pt', map_location=device_name)

    model.load_state_dict(last_model)

    optimizer.load_state_dict(ckpt['optimizer_state_dict'])
    for state in optimizer.state.values():
        for k, v in state.items():
            if torch.is_tensor(v):
                state[k] = v.to(torch.device(device_name))

    if scheduler is not None:
        scheduler.load_state_dict(ckpt['scheduler_state_dict'])

    start = ckpt['epoch']
    best_acc = ckpt['best_acc']
    epoch_data = ckpt['epoch_data']

    return start, best_acc, model, optimizer, scheduler, epoch_data
