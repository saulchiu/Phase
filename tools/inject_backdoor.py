import PIL.Image
import os
import pywt

import sys

import cv2
import numpy
import numpy as np

sys.path.append('../')
from tools.img import rgb2yuv, yuv2rgb, tensor2ndarray, ndarray2tensor, dct_2d_3c_slide_window, idct_2d_3c_slide_window, \
    idct_2d_3c_slide_window, dct_2d_3c_slide_window, dwt_2d_3c, idwt_2d_3c, fft_2d_3c, ifft_2d_3c
from tools.ctrl_transform import ctrl
from tools.img import rgb_to_yuv, yuv_to_rgb

import torch
import torchvision
from torchvision.transforms.transforms import ToTensor, Resize, Compose
from PIL import Image
import torch.nn.functional as F
from omegaconf import OmegaConf, DictConfig
import random


def patch_trigger(x_0: torch.Tensor, attack_name: str, config: DictConfig=None) -> torch.Tensor:
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
        return torch.clip(x_0, 0, 1)
    elif attack_name == 'ftrojan':
        channel_list = [1, 2]
        window_size = 32
        pos_list = [(15, 15), (31, 31)]
        magnitude = 30
        x_0 = tensor2ndarray(x_0)
        x_0 = rgb2yuv(x_0)
        x_dct = dct_2d_3c_slide_window(x_0, window_size)
        x_dct = np.transpose(x_dct, (2, 0, 1))
        for ch in channel_list:
            for w in range(0, x_dct.shape[1], window_size):
                for h in range(0, x_dct.shape[2], window_size):
                    for pos in pos_list:
                        x_dct[ch][w + pos[0], h + pos[1]] += magnitude
        x_dct = np.transpose(x_dct, (1, 2, 0))
        x_idct = idct_2d_3c_slide_window(x_dct, window_size)
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
    elif attack_name == 'duba':
        x_c = tensor2ndarray(x_0) / 255.  # scale to 0~1
        x_p_img = PIL.Image.open('../resource/example/000001.jpg')
        wave = 'db2'
        alpha = beta = 0.4

        # do 3-level dwt for x_c
        x_c_re = np.zeros_like(x_c)
        for i in range(3):  # 3 channel
            L_1, (H_11, H_12, H_13) = pywt.dwt2(x_c[:, :, i], wave)
            L_2, (H_21, H_22, H_23) = pywt.dwt2(L_1, wave)
            L_3, (H_31, H_32, H_33) = pywt.dwt2(L_2, wave)

            x_p1 = np.array(x_p_img.resize((L_1.shape[0], L_1.shape[1]))) / 255.
            _, (HP_21, HP_22, HP_23) = pywt.dwt2(x_p1[:, :, i], wave)
            x_p2 = np.array(x_p_img.resize((L_2.shape[0], L_2.shape[1]))) / 255.
            _, (HP_31, HP_32, HP_33) = pywt.dwt2(x_p2[:, :, i], wave)

            H_21 = beta * H_21 + (1 - beta) * HP_21
            H_22 = beta * H_22 + (1 - beta) * HP_22
            H_23 = beta * H_23 + (1 - beta) * HP_23

            H_31 = alpha * H_31 + (1 - alpha) * HP_31
            H_32 = alpha * H_32 + (1 - alpha) * HP_32
            H_33 = alpha * H_33 + (1 - alpha) * HP_33

            res_coe = [L_3, (H_31, H_32, H_33), (H_21, H_22, H_23), (H_11, H_12, H_13)]
            x_c_re[:, :, i] = pywt.waverec2(coeffs=res_coe, wavelet=wave)

        # exchange amplitude
        x_c_f = fft_2d_3c(x_c)
        x_re_f = fft_2d_3c(x_c_re)
        clean_amplitude, clean_phase = np.abs(x_c_f), np.angle(x_c_f)
        poison_amplitude, poison_phase = np.abs(x_re_f), np.angle(x_re_f)
        x_re_f = clean_amplitude * np.exp(1j * poison_phase)
        x_re = ifft_2d_3c(x_re_f).real

        # blend DCT frequency
        lamb = 0.7
        x_re_dct_1 = dct_2d_3c_slide_window(x_re.astype(float))
        x_c_dct_1 = dct_2d_3c_slide_window(x_c.astype(float))
        x_re_dct_2 = dct_2d_3c_slide_window(x_re_dct_1.astype(float))
        x_c_dct_2 = dct_2d_3c_slide_window(x_c_dct_1.astype(float))
        x_re_dct_2 = lamb * x_re_dct_2 + (1 - lamb) * x_c_dct_2
        x_re_dct_1 = idct_2d_3c_slide_window(x_re_dct_2)
        x_re_dct_1 = lamb * x_re_dct_1 + (1 - lamb) * x_c_dct_1
        x_re = idct_2d_3c_slide_window(x_re_dct_1)

        x_re = np.clip(x_re, 0, 1)
        return ndarray2tensor(x_re * 255.)
    elif attack_name == 'inba':
        # x = tensor2ndarray(x_0)
        # wind = 32
        # x_yuv = rgb2yuv(x)
        # x_fft = np.fft.fft2(x_yuv[:, :, 1])
        # imag_part = x_fft.imag
        # tg: torch.tensor = torch.load('../results/gtsrb/inba/20240928121249/trigger.pth')["tg_after"]
        # imag_part[0:wind, 0:wind] = tg.detach().numpy()
        # x_fft = x_fft.real + imag_part * 1j
        # x_yuv[:, :, 1] = np.fft.ifft2(x_fft).real
        # x_re = yuv2rgb(x_yuv)
        # x_re = ndarray2tensor(x_re)
        # x_re = torch.clip(x_re, 0, 1)
        # return x_re
        x_torch = x_0.detach().clone()
        x_torch *= 255.
        x_yuv = torch.stack(rgb_to_yuv(x_torch[0], x_torch[1], x_torch[2]), dim=0)
        x_yuv = torch.clip(x_yuv, 0, 255)
        tg: torch.tensor = torch.load('../results/gtsrb/inba/20240928135130/trigger.pth')["tg_after"]

        # inject trigger
        tg_size = config.attack.wind
        tg_pos = random.randint(0, tg_size)
        x_fft = torch.fft.fft2(x_yuv[1])
        x_imag = torch.imag(x_fft)
        x_imag[tg_pos:(tg_pos + tg_size), tg_pos:(tg_pos + tg_size)] = tg
        x_fft = torch.real(x_fft) + 1j * x_imag
        x_yuv[1] = torch.real(torch.fft.ifft2(x_fft))

        x_rgb = torch.stack(yuv_to_rgb(x_yuv[0], x_yuv[1], x_yuv[2]), dim=0)
        x_rgb = torch.clip(x_rgb, 0, 255)
        x_rgb /= 255.
        return x_rgb
    else:
        raise NotImplementedError(attack_name)
