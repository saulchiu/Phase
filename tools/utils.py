# universal function
import sys

REPO_ROOT = '/home/chengyiqiu/code/INBA/'

sys.path.append(REPO_ROOT)

import torch
import numpy as np
import random
import time
import os

def manual_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_model(name, num_class, device):
    if name == "resnet18":
        from classifier_models.resnet import ResNet18
        net = ResNet18(num_classes=num_class).to(device)
    elif name == "convnext":
        # lr=1e-3 weight_decay=1e-1
        from classifier_models.convnext import ConvNeXt
        net = ConvNeXt(num_class,
                        channel_list = [64, 128, 256, 512],
                        num_blocks_list = [2, 2, 2, 2],
                        kernel_size=7, patch_size=1,
                        res_p_drop=0.)
    elif name == "repvit":
        from classifier_models.repvit import repvit_m1_0 as RepViT
        net = RepViT(num_classes = num_class)
    # for defense, we use the same models as their original repositories.
    elif name == "rnp":
        from classifier_models.defense.RNP_model import resnet18
        net = resnet18(num_classes=num_class).to(device)
    elif name == "fp":
        from classifier_models.defense.fp.preact_resnet import PreActResNet18
        net = PreActResNet18(num_classes=num_class).to(device)
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