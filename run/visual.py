import sys
sys.path.append('../')
from models.cnn_lightning_model import BASELightningModule
from models.preact_resnet import PreActResNet18
import os
import torch


def get_lightning_checkpoints(res_path):
    checkpoint_folder = os.path.join(res_path, 'log', 'version_0', 'checkpoints')
    ckpt_files = [f for f in os.listdir(checkpoint_folder) if f.endswith('.ckpt')]

    if ckpt_files:
        print(f"Found ckpt file: {ckpt_files[0]}")
    else:
        print("No ckpt files found.")

    return ckpt_files[0]

res_path = '../results/imagenette/inba/20240926233859'
lr = 1e-2
momentum = 0.9
weight_decay = 5e-4
ckpt_name = get_lightning_checkpoints(res_path)

net = PreActResNet18(num_classes=10).to('cuda:0')
# model = MyLightningModule(net, lr, momentum, weight_decay)
dic = torch.load(f'{res_path}/log/version_0/checkpoints/{ckpt_name}')
model = BASELightningModule.load_from_checkpoint(f"{res_path}/log/version_0/checkpoints/{ckpt_name}")

