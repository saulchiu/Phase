import PIL.Image
import os
import pywt

import sys

import cv2
import numpy
import numpy as np

sys.path.append('../')
from tools.img import rgb2yuv, yuv2rgb, tensor2ndarray, ndarray2tensor, dct_2d_3c_slide_window, idct_2d_3c_slide_window, \
    idct_2d_3c_slide_window, dct_2d_3c_slide_window, fft_2d_3c, ifft_2d_3c
from tools.ctrl_transform import ctrl
from tools.img import rgb_to_yuv, yuv_to_rgb


import torch
import torchvision
from torchvision.transforms.transforms import ToTensor, Resize, Compose
from PIL import Image
import torch.nn.functional as F
from omegaconf import OmegaConf, DictConfig
import random
import math
import pywt


class BadTransform(object):
    def __init__(self, config) -> None:
        self.config = config

    def __call__(self, x_c: torch.tensor):
        # Attacker should decide whether apply normalization in somethere outside this function.
        x_p = patch_trigger(x_c, self.config)
        return x_p


def patch_trigger(x_0: torch.Tensor, config) -> torch.Tensor:  # do not do any clip operation here.
    """
    add a trigger to the original image given attack method
    :param x_0:
    :param attack_name: e.g., noise, badnet, blended, ftrojan, lf (low frequency), ctrl, wanet
    :return: poison image with trigger
    """
    attack_config = config.attack
    attack_name = attack_config.name
    c, h, w = x_0.shape
    trans = Compose([ToTensor(), Resize((h, h))])
    device = x_0.device
    if attack_name == "benign":
        x_p = x_0
    elif attack_name == 'blended':
        tg = Image.open(attack_config.tg_path)
        tg = trans(tg)
        tg = tg.to(x_0.device)
        x_p = x_0 * (1 - attack_config.blended_coeff) + tg * attack_config.blended_coeff
    elif attack_name == 'badnet':
        tg = Image.open(f'{attack_config.tg_path}/trigger_{h}_{int(h / 10)}.png')
        mask = Image.open(f'{attack_config.tg_path}/mask_{h}_{int(h / 10)}.png')
        tg = trans(tg)
        mask = trans(mask)
        tg = tg.to(x_0.device)
        mask = mask.to(x_0.device)
        x_p = (1 - mask) * x_0 + tg * mask
    elif attack_name == 'noise':
        factor = 0.05
        x_p = (1 - factor) * x_0 + factor * torch.randn_like(x_0, device=x_0.device)
        x_p = torch.clip(x_p, 0, 1)
    elif attack_name == 'ftrojan':
        channel_list = [1, 2]
        window_size = 32
        pos_list = [(15, 15), (31, 31)]
        if config.dataset_name in ["cifar10", "gtsrb"]:
            magnitude = 30
        else:
            magnitude = 50
        x_p = tensor2ndarray(x_0)
        x_p = rgb2yuv(x_p)
        x_dct = dct_2d_3c_slide_window(x_p, window_size)
        x_dct = np.transpose(x_dct, (2, 0, 1))
        for ch in channel_list:
            for w in range(0, x_dct.shape[1], window_size):
                for h in range(0, x_dct.shape[2], window_size):
                    for pos in pos_list:
                        x_dct[ch][w + pos[0], h + pos[1]] += magnitude
        x_dct = np.transpose(x_dct, (1, 2, 0))
        x_idct = idct_2d_3c_slide_window(x_dct, window_size)
        x_idct = yuv2rgb(x_idct)
        x_p = ndarray2tensor(x_idct)
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
        x_p = ndarray2tensor(x_0)
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
        grid_path = f'{config.path}/grid.pth'
        k = config.attack.k
        s = config.attack.s
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
        x_p = x_0
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
        x_p = trans(x_0)
    elif attack_name == 'duba':
        # x_c = tensor2ndarray(x_0) / 255.  # scale to 0~1
        x_c = tensor2ndarray(x_0)
        x_p_img = PIL.Image.open('../resource/DUBA/64.png')
        wave = 'db2'
        alpha = beta = config.attack.alpha

        # do 3-level dwt for x_c
        x_c_re = np.zeros_like(x_c)
        # for i in range(3):  # 3 channel
        #     L_1, (H_11, H_12, H_13) = pywt.dwt2(x_c[:, :, i], wave)
        #     L_2, (H_21, H_22, H_23) = pywt.dwt2(L_1, wave)
        #     L_3, (H_31, H_32, H_33) = pywt.dwt2(L_2, wave)

        #     x_p1 = np.array(x_p_img.resize((L_1.shape[0], L_1.shape[1])))
        #     _, (HP_21, HP_22, HP_23) = pywt.dwt2(x_p1[:, :, i], wave)
        #     x_p2 = np.array(x_p_img.resize((L_2.shape[0], L_2.shape[1])))
        #     _, (HP_31, HP_32, HP_33) = pywt.dwt2(x_p2[:, :, i], wave)

        #     H_21 = beta * H_21 + (1 - beta) * HP_21
        #     H_22 = beta * H_22 + (1 - beta) * HP_22
        #     H_23 = beta * H_23 + (1 - beta) * HP_23

        #     H_31 = alpha * H_31 + (1 - alpha) * HP_31
        #     H_32 = alpha * H_32 + (1 - alpha) * HP_32
        #     H_33 = alpha * H_33 + (1 - alpha) * HP_33

        #     res_coe = [L_3, (H_31, H_32, H_33), (H_21, H_22, H_23), (H_11, H_12, H_13)]
        #     x_c_re[:, :, i] = pywt.waverec2(coeffs=res_coe, wavelet=wave)
        
        [cl,(cH3,cV3,cD3),(cH2,cV2,cD2),(cH1,cV1,cD1)] = pywt.wavedec2(x_c, wavelet=wave, level=3, axes=(0, 1))
        # get the resize shape
        L1, (_, _, _) = pywt.wavedec2(x_c, wavelet=wave, level=1, axes=(0, 1))
        L2, (_, _, _), (_, _, _) = pywt.wavedec2(x_c, wavelet=wave, level=2, axes=(0, 1))
        x_p = np.array(x_p_img.resize((h, w)))
        x_p1 = np.array(x_p_img.resize((L1.shape[0], L1.shape[1])))
        x_p2 = np.array(x_p_img.resize((L2.shape[0], L2.shape[1])))
        [_, (ch1, cv1, cd1)] = pywt.wavedec2(x_p2, 'db2', level=1, axes=(0, 1))
        # [_, (chb1, cvb1, cdb1)] = pywt.wavedec2(x_p1, 'db2', level=1, axes=(0, 1))
        [cb1, (chb1, cvb1, cdb1),(chb2, cvb2, cdb2)] = pywt.wavedec2(x_p, wave, level=2, axes=(0, 1))       

        cH3 = cH3 * alpha + ch1 * (1 - alpha)
        cV3 =  cV3 * alpha + cv1 * (1 - alpha)
        cD3 =  cD3 * alpha + cd1 * (1 - alpha)
        # cH3 = cH3 + ch1 * (1 - alpha)
        # cV3 =  cv1 * (1 - alpha)
        # cD3 =  cd1 * (1 - alpha)

        cH2 = cH2 * beta + chb1 * (1 - beta)
        cV2 =  cV2 * beta + cvb1 * (1 - beta)
        cD2 = cD2 * beta + cdb1 * (1 - beta)
        # cH2 = cH2 + chb1 * (1 - beta)
        # cV2 =  cvb1 * (1 - beta)
        # cD2 = cdb1 * (1 - beta)
        x_c_re = pywt.waverec2([cl,(cH3,cV3,cD3),(cH2,cV2,cD2),(cH1,cV1,cD1)], wave, axes=(0, 1))
        x_c_re = np.array(x_c_re,np.float32)

        # exchange amplitude
        x_c_f = np.fft.fft2(x_c, axes=(-2, -3))
        x_re_f = np.fft.fft2(x_c_re, axes=(-2, -3))
        clean_amplitude, clean_phase = np.abs(x_c_f), np.angle(x_c_f)
        poison_amplitude, poison_phase = np.abs(x_re_f), np.angle(x_re_f)
        clean_amplitude = np.fft.fftshift(clean_amplitude, axes=(0, 1))
        clean_amplitude = np.fft.ifftshift(clean_amplitude, axes=(0, 1))
        x_re_f = clean_amplitude * np.exp(1j * poison_phase)
        x_re = np.fft.ifft2(x_re_f, axes=(-2, -3)).real

        # blend DCT frequency
        lamb = config.attack.lamb 
        x_re_dct_1 = dct_2d_3c_slide_window(x_re.astype(float))
        x_c_dct_1 = dct_2d_3c_slide_window(x_c.astype(float))
        x_re_dct_2 = dct_2d_3c_slide_window(x_re_dct_1.astype(float))
        x_c_dct_2 = dct_2d_3c_slide_window(x_c_dct_1.astype(float))
        x_re_dct_2 = lamb * x_re_dct_2 + (1 - lamb) * x_c_dct_2
        x_re_dct_1 = idct_2d_3c_slide_window(x_re_dct_2)
        x_re_dct_1 = lamb * x_re_dct_1 + (1 - lamb) * x_c_dct_1
        x_re = idct_2d_3c_slide_window(x_re_dct_1)

        # x_re = np.clip(x_re, 0, 1)
        # mask trigger
        # x_re *= 255.
        # x_c *= 255.
        x_re[x_c >= 220] = x_c [x_c >= 220]
        x_re[x_c <= 30] = x_c[x_c <= 30]
        x_re = x_re.astype(np.float32)
        x_p = ndarray2tensor(x_re)
    elif attack_name == 'inba':
        # ld = torch.load(f'{config.path}/trigger.pth')
        # u_tg = ld['u_tg'].to(device)
        # v_tg = ld['v_tg'].to(device)

        x_p = x_0.clone()
        x_yuv = torch.stack(rgb_to_yuv(x_p[0], x_p[1], x_p[2]), dim=0)
        u_pi_coeff = config.attack.u_pi_coeff
        v_pi_coeff = config.attack.v_pi_coeff
        v_size_coeff = config.attack.v_size_coeff
        
        # Y channel
        # x_y = x_yuv[0]
        # x_y_fft = torch.fft.fft2(x_y)
        # x_y_fft_real = torch.real(x_y_fft)
        # x_y_fft_imag = torch.imag(x_y_fft) * m
        # x_y_fft = x_y_fft_real + 1j * x_y_fft_imag
        # x_y = torch.real(torch.fft.ifft2(x_y_fft))
        # x_yuv[0] = x_y

        # U channel
        scale = x_0.shape[-1]
        tg_pos = int(scale / 2)
        tg_size = config.attack.tg_size
        x_u = x_yuv[1]
        x_u_fft = torch.fft.fft2(x_u)
        x_u_fft_amp = torch.abs(x_u_fft)
        x_u_fft_pha = torch.angle(x_u_fft)
        x_u_fft_pha[tg_pos-tg_size:tg_pos+tg_size, tg_pos-tg_size:tg_pos+tg_size] = math.pi * u_pi_coeff
        x_u_fft = x_u_fft_amp * torch.exp(1j * x_u_fft_pha)
        x_u = torch.fft.ifft2(x_u_fft)
        x_u = torch.real(x_u)
        x_yuv[1] = x_u

        # V channel
        tg_size = int(tg_size * v_size_coeff)
        x_v = x_yuv[2]
        x_v_fft = torch.fft.fft2(x_v)
        x_v_fft_amp = torch.abs(x_v_fft)
        x_v_fft_pha = torch.angle(x_v_fft)
        x_v_fft_pha[tg_pos-tg_size:tg_pos+tg_size, tg_pos-tg_size:tg_pos+tg_size] = math.pi * v_pi_coeff
        x_v_fft = x_v_fft_amp * torch.exp(1j * x_v_fft_pha)
        x_v = torch.fft.ifft2(x_v_fft)
        x_v = torch.real(x_v)
        x_yuv[2] = x_v
        
        x_p = torch.stack(yuv_to_rgb(x_yuv[0], x_yuv[1], x_yuv[2]), dim=0)

        x_c = x_0.clone()
        # mix amp
        x_c_fft = torch.fft.fft2(x_c, dim=(1, 2))
        x_p_fft = torch.fft.fft2(x_p, dim=(1, 2))
        x_p_fft = torch.abs(x_c_fft) * torch.exp(1j * torch.angle(x_p_fft))
        x_p = torch.fft.ifft2(x_p_fft, dim=(1, 2))

        x_p = torch.real(x_p)

    else:
        raise NotImplementedError(attack_name)
    x_p = x_p.to(x_0.device)
    return x_p
