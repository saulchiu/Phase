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
from tools.dataset import get_dataset_normalization
from tools.inject_backdoor import patch_trigger
from tools.dataset import get_dataloader
import torch.nn.functional as F
from models.cnn_lightning_model import INBALightningModule
import hydra
from omegaconf import DictConfig, OmegaConf
from tools.time import now
import os
import shutil
import yaml
from pytorch_lightning.loggers import CSVLogger
from tools.img import rgb_to_yuv, yuv_to_rgb
from skimage.metrics import structural_similarity
from tools.dataset import get_de_normalization, get_dataset_normalization

_ = torch.manual_seed(42)


@hydra.main(version_base=None, config_path='../config', config_name='default')
def train_mdoel(config: DictConfig):
    ratio = config.ratio
    dataset_name = config.dataset_name
    attack_name = config.attack.name
    target_label = config.target_label
    nw = config.num_workers
    epoch = config.epoch
    lr = config.lr
    momentum = config.momentum
    weight_decay = config.weight_decay
    wind = config.attack.wind
    device = config.device

    # save config, and source file
    print(OmegaConf.to_yaml(OmegaConf.to_object(config)))
    target_folder = f'../results/{dataset_name}/{attack_name}/{now()}' if config.path == 'None' else config.path
    if not os.path.exists(target_folder):
        os.makedirs(target_folder)
    train_target_path = os.path.join(target_folder, 'train.py')
    shutil.copy(__file__, train_target_path)
    train_target_path = os.path.join(target_folder, 'cnn_lightning_model.py')
    shutil.copy('../models/cnn_lightning_model.py', train_target_path)
    with open(f'{target_folder}/config.yaml', 'w') as f:
        yaml.dump(OmegaConf.to_object(config), f, allow_unicode=True)

    norm = get_dataset_normalization(dataset_name)
    if dataset_name == 'imagenette':
        batch = 64
        scale = 224
        num_classes = 10
        trans = Compose([ToTensor(), Resize((scale, scale)), norm])
        train_ds = torchvision.datasets.Imagenette(root='../data', split='train', transform=trans)
        test_ds = torchvision.datasets.Imagenette(root='../data', split='val', transform=trans)
    elif dataset_name == 'cifar10':
        num_classes = 10
        batch = 128
        scale = 32
        trans = Compose([ToTensor(), Resize((scale, scale)), norm])
        train_ds = torchvision.datasets.CIFAR10(root='../data', train=True, transform=trans)
        test_ds = torchvision.datasets.CIFAR10(root='../data', train=False, transform=trans)
    elif dataset_name == 'gtsrb':  # bad performance
        num_classes = 43
        batch = 128
        scale = 32
        trans = Compose([ToTensor(), Resize((scale, scale)), norm])
        train_ds = torchvision.datasets.GTSRB(root='../data', split='train', transform=trans)
        test_ds = torchvision.datasets.GTSRB(root='../data', split='test', transform=trans)
    else:
        raise NotImplementedError(dataset_name)
    train_dl = DataLoader(dataset=train_ds, batch_size=batch, shuffle=True, num_workers=nw)
    test_dl = DataLoader(dataset=test_ds, batch_size=batch, shuffle=False, num_workers=nw)
    net = PreActResNet18(num_classes=num_classes).to(f'cuda:{device}')
    model = INBALightningModule(net, config)
    # tg_before = model.trigger
    tg_before = model.trigger.detach().clone()
    logger = CSVLogger(save_dir=target_folder, name='log')
    trainer = L.Trainer(max_epochs=epoch, devices=[device], logger=logger, default_root_dir=target_folder)
    trainer.fit(model=model, train_dataloaders=train_dl)
    model.eval()
    print('----------benign----------')
    trainer.test(model=model, dataloaders=test_dl)  # benign performance
    print('----------poison----------')
    poison_test_list = []
    for x, y in iter(test_dl):
        for i in range(x.shape[0]):
            if y[i] == target_label:
                continue
            # craft poison data
            x[i] = get_de_normalization(dataset_name)(x[i])
            # tg_pos = random.randint(0, wind)
            tg_pos = 0
            # x_rgb = x[i].detach().clone()
            x_p = x[i]
            x_p *= 255.
            x_yuv = torch.stack(rgb_to_yuv(x_p[0], x_p[1], x_p[2]), dim=0)
            x_yuv = torch.clip(x_yuv, 0, 255)
            x_u = x_yuv[1]
            x_u_fft = torch.fft.fft2(x_u)
            x_u_fft_imag = torch.imag(x_u_fft)
            x_u_fft_imag[tg_pos:(tg_pos+wind), tg_pos:(tg_pos+wind)] = model.trigger
            x_u_fft = torch.real(x_u_fft) + 1j * x_u_fft_imag
            x_u = torch.real(torch.fft.ifft2(x_u_fft))

            x_yuv[1] = x_u
            x_p = torch.stack(yuv_to_rgb(x_yuv[0], x_yuv[1], x_yuv[2]), dim=0)
            x_p = torch.clip(x_p, 0, 255)
            x_p /= 255.

            # import matplotlib.pylab as plt
            #
            # _, ax = plt.subplots(1, 2)
            # ax[0].imshow(tensor2ndarray(x_rgb).astype(np.uint8))
            # ax[1].imshow(tensor2ndarray(x_p).astype(np.uint8))
            # plt.show()
            # print(structural_similarity(tensor2ndarray(x_rgb).astype(np.uint8), tensor2ndarray(x_p).astype(np.uint8), win_size=3))
        
            x[i] = x_p
            y[i] = target_label
            poison_test_list.append((x[i], y[i]))
    print(len(poison_test_list))
    poison_test_dl = DataLoader(dataset=List2Dataset(poison_test_list), batch_size=batch, shuffle=False, num_workers=nw)
    trainer.test(model=model, dataloaders=poison_test_dl)  # poison performance
    torch.save({
        "tg_before": tg_before,
        "tg_after": model.trigger
    }, f'{target_folder}/trigger.pth')


if __name__ == '__main__':
    train_mdoel()
