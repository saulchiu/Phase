import sys

sys.path.append('../')
from models.preact_resnet import PreActResNet18
import torchvision
from torchvision.transforms.transforms import ToTensor, Resize, Compose, RandomCrop
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
from models.cnn_lightning_model import INBALightningModule, visualize_metrics
import hydra
from omegaconf import DictConfig, OmegaConf
from tools.time import now
import os
import shutil
import yaml
from pytorch_lightning.loggers import CSVLogger
from tools.dataset import get_dataset_class_and_scale
from torchvision.models.convnext import ConvNeXt, CNBlockConfig
from tools.utils import manual_seed
from tools.dataset import PoisonDataset, get_train_and_test_dataset

@hydra.main(version_base=None, config_path='../config', config_name='default')
def train_mdoel(config: DictConfig):
    assert config.attack.name == "inba"
    manual_seed(config.seed)
    # save config, and source file
    target_folder = f'../results/{config.dataset_name}/{config.attack.name}/{now()}' if config.path == 'None' else config.path
    if not os.path.exists(target_folder):
        os.makedirs(target_folder)
    config.path = target_folder
    train_target_path = os.path.join(target_folder, 'train.py')
    shutil.copy(__file__, train_target_path)
    train_target_path = os.path.join(target_folder, 'cnn_lightning_model.py')
    shutil.copy('../models/cnn_lightning_model.py', train_target_path)
    with open(f'{target_folder}/config.yaml', 'w') as f:
        yaml.dump(OmegaConf.to_object(config), f, allow_unicode=True)
    print(OmegaConf.to_yaml(OmegaConf.to_object(config)))

    train_dl, test_dl = get_dataloader(
        config.dataset_name,
        config.batch,
        pin_memory=config.pin_memory,
        num_workers=config.num_workers
    )
    num_classes, _ = get_dataset_class_and_scale(config.dataset_name)
    if config.model == "resnet18":
        net = PreActResNet18(num_classes=num_classes).to(f'cuda:{config.device}')
    elif config.model == "rnp":
        from models.resnet_cifar import resnet18
        net = resnet18(num_classes=num_classes).to(f'cuda:{config.device}')
    elif config.model == "repvgg":
        from repvgg_pytorch.repvgg import RepVGG
        net = RepVGG(num_blocks=[2, 4, 14, 1], num_classes=num_classes, width_multiplier=[1.5, 1.5, 1.5, 2.75]).to(device=f'cuda:{config.device}')
    else:
        raise NotImplementedError(config.model)
    model = INBALightningModule(net, config)
    # tg_before = model.trigger
    tg_before = model.trigger.detach().clone()
    logger = CSVLogger(save_dir=target_folder, name='log')
    assert config.epoch > config.val_epoch
    assert config.epoch > config.attack.tg_epoch
    trainer = L.Trainer(max_epochs=config.epoch + config.attack.tg_epoch, devices=[config.device], logger=logger, default_root_dir=target_folder)
    trainer.fit(model=model, train_dataloaders=train_dl)
    torch.save({
    "tg_before": tg_before,
    "tg_after": model.trigger
    },
    f'{target_folder}/trigger.pth')
    if config.model == "repvgg":
        model.model.deploy =True
    model.eval()
    print('----------benign----------')
    trainer.test(model=model, dataloaders=test_dl)  # benign performance
    # print('----------poison----------')
    # config_test = config.copy()
    # config_test.ratio = 1
    # _, test_ds = get_train_and_test_dataset(config.dataset_name)
    # poison_test_ds = PoisonDataset(test_ds, config_test)
    # poison_test_dl = DataLoader(poison_test_ds, batch_size=config.batch, shuffle=False, num_workers=config.num_workers, drop_last=False, pin_memory=config.pin_memory)
    # trainer.test(model=model, dataloaders=poison_test_dl)  # poison performance
    res = {
    "model": model.model.state_dict(),
    "param_opt": model.param_opt.state_dict(),
    "tg_opt": model.tg_opt.state_dict(),
    "schedule": model.scheduler.state_dict(),
    "config": config,
    "epoch": model.current_epoch,
    }
    torch.save(res, f"{target_folder}/results.pth")
    visualize_metrics(model.metrics_list, target_folder)



if __name__ == '__main__':
    train_mdoel()
