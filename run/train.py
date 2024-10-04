import sys
sys.path.append('../')
from models.preact_resnet import PreActResNet18
import torchvision
from torchvision.transforms.transforms import ToTensor, Resize, Compose
import torch
from torch.utils.data.dataloader import DataLoader
import random
from tools.img import tensor2ndarray, ndarray2tensor
from tools.img import yuv2rgb, rgb2yuv
from tools.img import fft_2d_3c, ifft_2d_3c
import pytorch_lightning as L
from tools.dataset import List2Dataset
import numpy as np
from tools.dataset import get_dataset_normalization, get_de_normalization
from tools.inject_backdoor import patch_trigger 
from tools.dataset import get_dataloader, get_benign_transform
import torch.nn.functional as F
import cv2
from models.cnn_lightning_model import BASELightningModule
import hydra
from omegaconf import DictConfig, OmegaConf
from tools.time import now
import os
import shutil
import yaml
from pytorch_lightning.loggers import CSVLogger
from tools.utils import manual_seed

@hydra.main(version_base=None, config_path='../config', config_name='default')
def train_mdoel(config: DictConfig):
    resume_train = config.path != "None"
    seed = config.seed
    manual_seed(seed)

    ratio = config.ratio
    dataset_name = config.dataset_name
    attack_name = config.attack.name
    target_label = config.target_label
    nw = config.num_workers
    epoch = config.epoch
    # save config, and source file
    target_folder = config.path if resume_train else f'../results/{dataset_name}/{attack_name}/{now()}'
    config.path = target_folder
    if not os.path.exists(target_folder):
        os.makedirs(target_folder)
    main_target_path = os.path.join(target_folder, 'train.py')
    shutil.copy(__file__, main_target_path)
    with open(f'{target_folder}/config.yaml', 'w') as f:
        yaml.dump(OmegaConf.to_object(config), f, allow_unicode=True)
    print(OmegaConf.to_yaml(OmegaConf.to_object(config)))

    batch = config.batch
    if dataset_name == 'imagenette':
        scale = 224
        num_classes = 10
        train_ds = torchvision.datasets.Imagenette(root='../data', split='train', transform=get_benign_transform(dataset_name, scale))
        test_ds = torchvision.datasets.Imagenette(root='../data', split='val', transform=get_benign_transform(dataset_name, scale, train=False))
    elif dataset_name == 'cifar10':
        num_classes = 10
        scale = 32
        train_ds = torchvision.datasets.CIFAR10(root='../data', train=True, transform=get_benign_transform(dataset_name, scale))
        test_ds = torchvision.datasets.CIFAR10(root='../data', train=False, transform=get_benign_transform(dataset_name, scale, train=False))
    elif dataset_name == 'gtsrb':
        num_classes = 43
        scale = 32
        train_ds = torchvision.datasets.GTSRB(root='../data', split='train', transform=get_benign_transform(dataset_name, scale))
        test_ds = torchvision.datasets.GTSRB(root='../data', split='test', transform=get_benign_transform(dataset_name, scale, train=False))
    elif dataset_name == 'fer2013':
        num_classes = 8
        scale = 64
        train_ds = torchvision.datasets.ImageFolder(root='../data/fer2013/train', transform=get_benign_transform(dataset_name, scale))
        test_ds = torchvision.datasets.ImageFolder(root='../data/fer2013/test', transform=get_benign_transform(dataset_name, scale, train=False))
    elif dataset_name == 'rafdb':
        num_classes = 7
        scale = 64
        train_ds = torchvision.datasets.ImageFolder(root='../data/RAF-DB/train', transform=get_benign_transform(dataset_name, scale))
        test_ds = torchvision.datasets.ImageFolder(root='../data/RAF-DB/test', transform=get_benign_transform(dataset_name, scale, train=False))
    else:
        raise NotImplementedError(dataset_name)
    train_dl = DataLoader(dataset=train_ds, batch_size=batch, shuffle=True, num_workers=nw)
    test_dl = DataLoader(dataset=test_ds, batch_size=batch, shuffle=False, num_workers=nw)
    poison_train_list = []
    for x, y in iter(train_dl):
        for i in range(x.shape[0]):
            if random.random() < ratio and attack_name != 'benign':  # craft poison data
                x_re = get_de_normalization(dataset_name)(x[i]).squeeze()
                # x_re = x[i]
                x_re = patch_trigger(x_re, config.attack)
                x[i] = x_re
                y[i] = target_label
            poison_train_list.append((x[i], y[i]))

    poison_test_list = []
    if attack_name != 'benign':
        for x, y in iter(test_dl):
            for i in range(x.shape[0]):
                if y[i] == target_label:
                    continue
                x_re = get_de_normalization(dataset_name)(x[i]).squeeze()
                # x_re = x[i]
                x_re = patch_trigger(x_re, config.attack)
                x[i] = x_re
                y[i] = target_label
                poison_test_list.append((x[i], y[i]))
        print(len(poison_train_list), len(poison_test_list))

    net = PreActResNet18(num_classes=num_classes).to('cuda:0')
    poison_train_dl = DataLoader(dataset=List2Dataset(poison_train_list), batch_size=batch, shuffle=True, num_workers=nw)
    model = BASELightningModule(net, config)
    logger = CSVLogger(save_dir=target_folder, name='log')
    trainer = L.Trainer(max_epochs=epoch, devices=[0], logger=logger, default_root_dir=target_folder, check_val_every_n_epoch=int(epoch / 2))
    trainer.fit(model=model, train_dataloaders=poison_train_dl, val_dataloaders=test_dl)
    print('----------benign----------')
    trainer.test(model=model, dataloaders=test_dl)  # benign performance
    if attack_name != 'benign':
        poison_test_dl = DataLoader(dataset=List2Dataset(poison_test_list), batch_size=batch, shuffle=False, num_workers=nw)
        print('----------poison----------')
        trainer.test(model=model, dataloaders=poison_test_dl)  # poison performance
    res = {
        "model": model.model.state_dict(),
        "ema": model.ema.state_dict(),
        "param_opt": model.opt.state_dict(),
        "schedule": model.schedule.state_dict(),
        "config": config,
        "epoch": model.current_epoch,
    }
    torch.save(res, f"{target_folder}/results.pth")




if __name__ == '__main__':
    train_mdoel()






