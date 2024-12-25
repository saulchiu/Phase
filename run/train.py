import os
CWD = os.getcwd()
REPO_ROOT = CWD.split('INBA')[0] + "INBA/"

import sys
sys.path.append(REPO_ROOT)
from torchvision.transforms.transforms import ToTensor, Resize, Compose
import torch
from torch.utils.data.dataloader import DataLoader
import pytorch_lightning as L
import numpy as np
from tools.inject_backdoor import patch_trigger 
from tools.dataset import get_dataloader, get_dataset_class_and_scale
import torch.nn.functional as F
from classifier_models.cnn_lightning_model import BASELightningModule, visualize_metrics
import hydra
from omegaconf import DictConfig, OmegaConf
from tools.time import now
import shutil
import yaml
from pytorch_lightning.loggers import CSVLogger
from tools.utils import manual_seed, get_model
from tools.dataset import PoisonDataset, get_dataloader, get_train_and_test_dataset


@hydra.main(version_base=None, config_path=f'{REPO_ROOT}/config', config_name='default')
def train_mdoel(config: DictConfig):
    manual_seed(config.seed)
    target_folder = config.path if config.path != "None" else f'{REPO_ROOT}/results/{config.dataset_name}/{config.attack.name}/{config.model}/{now()}'
    if not os.path.exists(target_folder):
        os.makedirs(target_folder)
    print(OmegaConf.to_yaml(OmegaConf.to_object(config)))
    print(target_folder)
    # save config, and source file
    config.path = target_folder
    main_target_path = os.path.join(target_folder, 'train.py')
    shutil.copy(__file__, main_target_path)
    train_target_path = os.path.join(target_folder, 'cnn_lightning_model.py')
    shutil.copy(f'{REPO_ROOT}/classifier_models/cnn_lightning_model.py', train_target_path)
    train_target_path = os.path.join(target_folder, 'inject_backdoor.py')
    shutil.copy(f'{REPO_ROOT}/tools/inject_backdoor.py', train_target_path)
    train_target_path = os.path.join(target_folder, 'dataset.py')
    shutil.copy(f'{REPO_ROOT}/tools/dataset.py', train_target_path)
    with open(f'{target_folder}/config.yaml', 'w') as f:
        yaml.dump(OmegaConf.to_object(config), f, allow_unicode=True)
    
    _, test_dl = get_dataloader(
        config.dataset_name,
        config.batch,
        config.pin_memory,
        config.num_workers
    )
    config_test = config.copy()
    config_test.ratio = 1
    train_ds, test_ds = get_train_and_test_dataset(config.dataset_name)
    poison_train_ds = PoisonDataset(train_ds, config)
    poison_test_ds = PoisonDataset(test_ds, config_test)
    num_classes, _ = get_dataset_class_and_scale(config.dataset_name)
    device = f'cuda:{config.device}' if config.device != 'cpu' else config.device
    net = get_model(config.model, num_classes, device=device)
    poison_train_dl = DataLoader(poison_train_ds, batch_size=config.batch, shuffle=True, num_workers=config.num_workers, drop_last=True, pin_memory=config.pin_memory)
    model = BASELightningModule(net, config)
    logger = CSVLogger(save_dir=target_folder, name='log')
    assert config.epoch >= config.val_epoch
    trainer = L.Trainer(max_epochs=config.epoch, devices=[config.device], logger=logger, default_root_dir=target_folder)
    trainer.fit(model=model, train_dataloaders=poison_train_dl)
    res = {
        "model": model.model.state_dict(),
        "param_opt": model.opt.state_dict(),
        "schedule": model.schedule.state_dict(),
        "config": config,
        "epoch": model.current_epoch,
    }
    torch.save(res, f"{target_folder}/results.pth")
    visualize_metrics(model.metrics_list, target_folder)
    print(OmegaConf.to_yaml(OmegaConf.to_object(config)))

    """
    evaluation
    """
    from run.eval_acc import cal_acc_asr
    from run.eval_ssim import cal_ssim_psnr
    b_acc, p_acc = cal_acc_asr(target_folder)
    ssim_metric, psnr_metric, lpips_metric = cal_ssim_psnr(target_folder)
    file_path = f"{target_folder}/metric.txt"
    with open(file_path, 'w') as file:
        file.write(f"BA: {b_acc}\n")
        file.write(f"ASR: {p_acc}\n")
        file.write(f"SSIM: {ssim_metric}\n")
        file.write(f"PSNR: {psnr_metric}\n")
        file.write(f"LPIPS: {lpips_metric}\n")




if __name__ == '__main__':
    train_mdoel()






