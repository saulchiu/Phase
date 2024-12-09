# universal function
import sys

REPO_ROOT = '/home/chengyiqiu/code/INBA/'

sys.path.append(REPO_ROOT)

import torch
import numpy as np
import random
import time
import os

# _, term_width = os.popen("stty size", "r").read().split()
# term_width = int(term_width)
# TOTAL_BAR_LENGTH = 65.0
# last_time = time.time()
# begin_time = last_time
# def progress_bar(current, total, msg=None):
#     global last_time, begin_time
#     if current == 0:
#         begin_time = time.time()  # Reset for new bar.

#     cur_len = int(TOTAL_BAR_LENGTH * current / total)
#     rest_len = int(TOTAL_BAR_LENGTH - cur_len) - 1

#     sys.stdout.write(" [")
#     for i in range(cur_len):
#         sys.stdout.write("=")
#     sys.stdout.write(">")
#     for i in range(rest_len):
#         sys.stdout.write(".")
#     sys.stdout.write("]")

#     cur_time = time.time()
#     step_time = cur_time - last_time
#     last_time = cur_time
#     tot_time = cur_time - begin_time

#     L = []
#     if msg:
#         L.append(" | " + msg)

#     msg = "".join(L)
#     sys.stdout.write(msg)
#     for i in range(term_width - int(TOTAL_BAR_LENGTH) - len(msg) - 3):
#         sys.stdout.write(" ")

#     # Go back to the center of the bar.
#     for i in range(term_width - int(TOTAL_BAR_LENGTH / 2) + 2):
#         sys.stdout.write("\b")
#     sys.stdout.write(" %d/%d " % (current + 1, total))

#     if current < total - 1:
#         sys.stdout.write("\r")
#     else:
#         sys.stdout.write("\n")
#     sys.stdout.flush()

def manual_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_model(name, num_class, device):
    if name == "resnet18":
        print(sys.path)
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
        # lr=1e-3 weight_decay=0.025
        from classifier_models.repvit import repvit_m1_0 as RepViT
        net = RepViT(num_classes = num_class)
    elif name == "presnet18":
        print(sys.path)
        from classifier_models.preact_resnet import PreActResNet18
        net = PreActResNet18(num_classes=num_class).to(device)
    elif name == "rnp":
        from classifier_models.resnet_cifar import resnet18
        net = resnet18(num_classes=num_class).to(device)
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