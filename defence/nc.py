import torch
import numpy as np
from torch.nn import CrossEntropyLoss
import tqdm
import torchvision


def train(model, target_label, train_loader, param):
    print("Processing label: {}".format(target_label))

    width, height = param["image_size"]
    trigger = torch.rand((3, width, height), requires_grad=True)
    trigger = trigger.to(device).detach().requires_grad_(True)
    mask = torch.rand((width, height), requires_grad=True)
    mask = mask.to(device).detach().requires_grad_(True)

    Epochs = param["Epochs"]
    lamda = param["lamda"]

    min_norm = np.inf
    min_norm_count = 0

    criterion = CrossEntropyLoss()
    optimizer = torch.optim.Adam([{"params": trigger},{"params": mask}],lr=0.005)
    model.to(device)
    model.eval()

    for epoch in range(Epochs):
        norm = 0.0
        for images, _ in tqdm.tqdm(train_loader, desc='Epoch %3d' % (epoch)):
            optimizer.zero_grad()
            images = images.to(device)
            trojan_images = (1 - torch.unsqueeze(mask, dim=0)) * images + torch.unsqueeze(mask, dim=0) * trigger
            y_pred = model(trojan_images)
            y_target = torch.full((y_pred.size(0),), target_label, dtype=torch.long).to(device)
            loss = criterion(y_pred, y_target) + lamda * torch.sum(torch.abs(mask))
            loss.backward()
            optimizer.step()

            with torch.no_grad():
                torch.clip_(trigger, 0, 1)
                torch.clip_(mask, 0, 1)
                norm = torch.sum(torch.abs(mask))
        print("norm: {}".format(norm))

        if norm < min_norm:
            min_norm = norm
            min_norm_count = 0
        else:
            min_norm_count += 1

        if min_norm_count > 30:
            break

    return trigger.cpu(), mask.cpu()

def outlier_detection(l1_norm_list):

    consistency_constant = 1.4826  # if normal distribution
    median = np.median(l1_norm_list)
    mad = consistency_constant * np.median(np.abs(l1_norm_list - median))
    min_mad = np.abs(np.min(l1_norm_list) - median) / mad

    print('median: %f, MAD: %f' % (median, mad))
    print('anomaly index: %f' % min_mad)
    return min_mad


from omegaconf import OmegaConf
import sys
sys.path.append('../')
from tools.utils import manual_seed, get_model, rm_if_exist
from tools.dataset import get_dataset_class_and_scale, get_dataloader
import os

if __name__ == "__main__":
    target_folder = '/home/chengyiqiu/code/INBA/results/cifar10/inba/20241112193251'
    path = f'{target_folder}/config.yaml'
    config = OmegaConf.load(path)
    manual_seed(config.seed)
    device = f'cuda:{config.device}'
    num_class, scale = get_dataset_class_and_scale(config.dataset_name)
    net = get_model(config.model, num_class, device=device)
    ld = torch.load(f'{target_folder}/results.pth', map_location=device)
    net.load_state_dict(ld['model'])
    net.to(device)
    if config.model == "repvgg":
        net.deploy =True
    param = {
        "dataset": config.dataset_name,
        "Epochs": 10,
        "batch_size": 512,
        "lamda": 0.01,
        "num_classes": num_class,
        "image_size": (scale, scale)
    }
    train_loader, _ = get_dataloader(config.dataset_name, param['batch_size'], config.pin_memory, config.num_workers)
    norm_list = []
    
    rm_if_exist(f'{target_folder}/nc')
    os.makedirs(f'{target_folder}/nc', exist_ok=True)
    for label in range(param["num_classes"]):
        trigger, mask = train(net, label, train_loader, param)
        norm_list.append(mask.sum().item())
        torchvision.utils.save_image(mask, f'{target_folder}/nc/mask_{label}.png', normalize=True)
        torchvision.utils.save_image(trigger * mask, f'{target_folder}/nc/trigger_{label}.png', normalize=True)
    print(norm_list)
    anomaly_index = outlier_detection(norm_list)