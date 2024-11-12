# universal function
import sys
sys.path.append('../')

import torch
import numpy as np
import random


def manual_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_model(name, num_class, device):
    if name == "resnet18":
        from models.preact_resnet import PreActResNet18
        net = PreActResNet18(num_classes=num_class).to(device)
    elif name == "rnp":
        from models.resnet_cifar import resnet18
        net = resnet18(num_classes=num_class).to(device)
    elif name == "repvgg":
        from repvgg_pytorch.repvgg import RepVGG
        net = RepVGG(num_blocks=[2, 4, 14, 1], num_classes=num_class, width_multiplier=[1.5, 1.5, 1.5, 2.75]).to(device)
    elif name == "convnext2":
        from models.convnext2 import convnextv2_huge
        net = convnextv2_huge(num_classes=num_class)
    elif name == "convnext":
        from models.convnext import ConvNeXt
        net = ConvNeXt(num_class,
                        channel_list = [64, 128, 256, 512],
                        num_blocks_list = [2, 2, 2, 2],
                        kernel_size=7, patch_size=1,
                        res_p_drop=0.)
    else:
        raise NotImplementedError(name)
    return net


import os
import shutil

def rm_if_exist(folder):
    if os.path.exists(folder):
        shutil.rmtree(folder)
        print(f"Folder '{folder}' has been removed.")
    else:
        print(f"Folder '{folder}' does not exist, no action taken.")