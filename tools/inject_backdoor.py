import sys

import cv2
import numpy
import numpy as np

sys.path.append('.')
from tools.img import rgb2yuv, yuv2rgb, tensor2ndarray, ndarray2tensor, dct2, idct2

import torch
import torchvision
from torchvision.transforms.transforms import ToTensor, Resize, Compose
from PIL import Image


def patch_trigger(x_0: torch.Tensor, attack_name: str) -> torch.Tensor:
    """
    add a trigger to the original image given attack method
    :param x_0:
    :param attack_name:
    :return: poison image with trigger
    """
    c, h, w = x_0.shape
    trans = Compose([ToTensor(), Resize((h, h))])
    if attack_name == 'blended':
        tg = Image.open('../resource/blended/hello_kitty.jpeg')
        tg = trans(tg)
        x_0 = x_0 * 0.8 + tg * 0.2
    elif attack_name == 'badnet':
        tg = Image.open(f'../resource/badnet/trigger_{h}_3.png')
        mask = Image.open(f'../resource/badnet/mask_{h}_3.png')
        tg = trans(tg)
        mask = trans(mask)
        x_0 = (1 - mask) * x_0 + tg * mask
    elif attack_name == 'noise':
        factor = 0.01
        x_0 = (1 - factor) * x_0 + factor * torch.randn_like(x_0, device=x_0.device)
    elif attack_name == 'ftrojan':
        channel_list = [1, 2]
        window_size = 32
        pos_list = [(15, 15), (31, 31)]
        magnitude = 30
        x_0 = tensor2ndarray(x_0)
        x_0 = rgb2yuv(x_0)
        x_0 = dct2(x_0)
        # for ch in channel_list:
        #     for w in range(0, x_0.shape[1], window_size):
        #         for h in range(0, x_0.shape[2], window_size):
        #             for pos in pos_list:
        #                 x_0[ch][w + pos[0], h + pos[1]] += magnitude
        x_0 = idct2(x_0)
        x_0 = yuv2rgb(x_0)
        x_0 = ndarray2tensor(x_0)
    else:
        raise NotImplementedError(attack_name)
    return x_0
