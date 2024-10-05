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
from tools.dataset import get_dataloader, get_benign_transform, get_poison_transform
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
from tools.dataset import PoisonDataset
from tools.inject_backdoor import BadTransform
import matplotlib.pyplot as plt
from pytorch_lightning.loggers import TensorBoardLogger
from repvgg_pytorch.repvgg import RepVGG
from torchvision.models.convnext import ConvNeXt, CNBlockConfig


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
        config_test = config.copy()
        config_test.ratio = 1
        poison_train_ds = PoisonDataset(train_ds, config)
        poison_test_ds = PoisonDataset(train_ds, config_test)
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
    train_dl = DataLoader(dataset=train_ds, batch_size=batch, shuffle=True, num_workers=nw, drop_last=True, pin_memory=config.pin_memory)
    test_dl = DataLoader(dataset=test_ds, batch_size=batch, shuffle=False, num_workers=nw, drop_last=False, pin_memory=config.pin_memory)

    if config.model == "resnet18":
        net = PreActResNet18(num_classes=num_classes).to(f'cuda:{0}')
    elif config.model == "repvgg":
        net = RepVGG(num_blocks=[4, 6, 16, 1], num_classes=num_classes, width_multiplier=[3, 3, 3, 5]).to(device=f'cuda:{0}')
    elif config.model == "convnext":
        if dataset_name == 'cifar10':
            channel_list = [96, 192, 384, 768]
            stochastic_depth_prob = 0.1
        elif dataset_name == 'imagenette':
            channel_list = [96, 192, 384, 768]
            stochastic_depth_prob = 0.2
        elif dataset_name == 'gtsrb':
            channel_list = [96, 192, 384, 768]
            stochastic_depth_prob = 0.3
        elif dataset_name == 'fer2013':
            channel_list = [64, 128, 256, 512]
            stochastic_depth_prob = 0.1
        else:
            raise NotImplementedError(dataset_name)
        block_setting = [
            CNBlockConfig(input_channels=channel_list[0], out_channels=channel_list[1], num_layers=3),
            CNBlockConfig(input_channels=channel_list[1], out_channels=channel_list[2], num_layers=3),
            CNBlockConfig(input_channels=channel_list[2], out_channels=channel_list[3], num_layers=9),
            CNBlockConfig(input_channels=channel_list[3], out_channels=None, num_layers=3)
        ]
        net = ConvNeXt(
            block_setting=block_setting,
            stochastic_depth_prob=stochastic_depth_prob,  # Lower stochastic depth for a small dataset
            layer_scale=1e-6,
            num_classes=num_classes
        ).to(f'cuda:{0}')
    else:
        raise NotImplementedError(config.model)

    poison_train_dl = DataLoader(poison_train_ds, batch_size=batch, shuffle=True, num_workers=nw, drop_last=True, pin_memory=config.pin_memory)
    model = BASELightningModule(net, config)
    # logger = TensorBoardLogger(save_dir=target_folder)
    logger = CSVLogger(save_dir=target_folder, name='log')
    assert config.epoch >= config.val_epoch
    trainer = L.Trainer(max_epochs=epoch, devices=[0], logger=logger, default_root_dir=target_folder)
    trainer.fit(model=model, train_dataloaders=poison_train_dl)
    print('----------benign----------')
    trainer.test(model=model, dataloaders=test_dl)  # benign performance
    if attack_name != 'benign':
        poison_test_dl = DataLoader(poison_test_ds, batch_size=batch, shuffle=False, num_workers=nw, drop_last=False, pin_memory=config.pin_memory)
        print('----------poison----------')
        trainer.test(model=model, dataloaders=poison_test_dl)  # poison performance
    res = {
        "model": model.model.state_dict(),
        "param_opt": model.opt.state_dict(),
        "schedule": model.schedule.state_dict(),
        "config": config,
        "epoch": model.current_epoch,
    }
    torch.save(res, f"{target_folder}/results.pth")
    # visualize_metrics(model.metrics_list, target_folder)




if __name__ == '__main__':
    train_mdoel()






