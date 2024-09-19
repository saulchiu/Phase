import os

import sys

import cv2
import numpy
import numpy as np

sys.path.append('../')
from tools.img import rgb2yuv, yuv2rgb, tensor2ndarray, ndarray2tensor, dct_2d_slide_window, idct_2d_slide_window, \
    idct_2d_slide_window, dct_2d_slide_window
from tools.ctrl_transform import ctrl

import torch
import torchvision
from torchvision.transforms.transforms import ToTensor, Resize, Compose
from PIL import Image
import torch.nn.functional as F


def patch_trigger(x_0: torch.Tensor, attack_name: str) -> torch.Tensor:
    """
    add a trigger to the original image given attack method
    :param x_0:
    :param attack_name: e.g., noise, badnet, blended, ftrojan, lf (low frequency), ctrl, wanet
    :return: poison image with trigger
    """
    c, h, w = x_0.shape
    trans = Compose([ToTensor(), Resize((h, h))])
    if attack_name == 'blended':
        tg = Image.open('../resource/blended/hello_kitty.jpeg')
        tg = trans(tg)
        x_0 = x_0 * 0.8 + tg * 0.2
        return x_0
    elif attack_name == 'badnet':
        tg = Image.open(f'../resource/badnet/trigger_{h}_3.png')
        mask = Image.open(f'../resource/badnet/mask_{h}_3.png')
        tg = trans(tg)
        mask = trans(mask)
        x_0 = (1 - mask) * x_0 + tg * mask
        return x_0
    elif attack_name == 'noise':
        factor = 0.05
        x_0 = (1 - factor) * x_0 + factor * torch.randn_like(x_0, device=x_0.device)
        return x_0
    elif attack_name == 'ftrojan':
        channel_list = [1, 2]
        window_size = 32
        pos_list = [(15, 15), (31, 31)]
        magnitude = 30
        x_0 = tensor2ndarray(x_0)
        x_0 = rgb2yuv(x_0)
        x_dct = dct_2d_slide_window(x_0, window_size)
        x_dct = np.transpose(x_dct, (2, 0, 1))
        for ch in channel_list:
            for w in range(0, x_dct.shape[1], window_size):
                for h in range(0, x_dct.shape[2], window_size):
                    for pos in pos_list:
                        x_dct[ch][w + pos[0], h + pos[1]] += magnitude
        x_dct = np.transpose(x_dct, (1, 2, 0))
        x_idct = idct_2d_slide_window(x_dct, window_size)
        x_idct = yuv2rgb(x_idct)
        return ndarray2tensor(x_idct)
    elif attack_name == 'lf':  # low frequency
        dataset_name = 'cifar10'
        model_name = 'preactresnet18'
        resource_path = f"../resource/lowFrequency/{dataset_name}_{model_name}_0_255.npy"
        trigger_array = np.load(resource_path)
        if len(trigger_array.shape) == 4:
            trigger_array = trigger_array[0]
        elif len(trigger_array.shape) == 3:
            pass
        elif len(trigger_array.shape) == 2:
            trigger_array = np.stack((trigger_array,) * 3, axis=-1)
        else:
            raise ValueError("lowFrequency trigger shape error, should be either 2 or 3 or 4")
        x_0 = tensor2ndarray(x_0)
        np.clip(x_0.astype(float) + trigger_array, 0, 255).astype(np.uint8)
        return ndarray2tensor(x_0)
    elif attack_name == 'wanet':
        def get_wanet_grid(image_size, grid_path: str, s: float, device='cpu'):
            grid = torch.load(grid_path)
            noise_grid = grid['noise_grid']
            identity_grid = grid['identity_grid']
            grid_temps = grid['grid_temps']
            noise_grid = noise_grid.to(device)
            identity_grid = identity_grid.to(device)
            grid_temps = grid_temps.to(device)
            assert torch.equal(grid_temps, torch.clamp(identity_grid + s * noise_grid / image_size * 1, -1, 1))
            return grid_temps

        image_size = x_0.shape[1]
        device = x_0.device
        grid_path = f'../resource/wanet/grid_{image_size}.pth'
        k = 4
        s = 0.5
        if os.path.exists(grid_path):
            grid_temps = get_wanet_grid(image_size, grid_path, s)
        else:
            ins = torch.rand(1, 2, k, k) * 2 - 1
            ins = ins / torch.mean(torch.abs(ins))
            noise_grid = (F.upsample(ins, size=image_size, mode="bicubic", align_corners=True)
                          .permute(0, 2, 3, 1).to(device))
            array1d = torch.linspace(-1, 1, steps=image_size)
            x, y = torch.meshgrid(array1d, array1d)
            identity_grid = torch.stack((y, x), 2)[None, ...].to(device)
            grid_temps = (identity_grid + s * noise_grid / image_size) * 1
            grid_temps = torch.clamp(grid_temps, -1, 1)
            grid = {
                'grid_temps': grid_temps,
                'noise_grid': noise_grid,
                'identity_grid': identity_grid,
            }
            torch.save(grid, grid_path)
        x_0 = x_0.unsqueeze(0)
        x_0 = F.grid_sample(x_0, grid_temps.repeat(1, 1, 1, 1), align_corners=True)
        x_0 = x_0.squeeze()
        return x_0
    elif attack_name == 'ctrl':
        class Args:
            pass

        args = Args()
        args.__dict__ = {
            "img_size": (x_0.shape[1], x_0.shape[1], 3),
            "use_dct": False,
            "use_yuv": True,
            "pos_list": [15, 31],
            "trigger_channels": (1, 2),
        }
        bad_transform = ctrl(args, False)
        x_0 = bad_transform(Image.fromarray(tensor2ndarray(x_0)), 1)
        return trans(x_0)
    elif attack_name == 'fiba':
        pass
    else:
        raise NotImplementedError(attack_name)
