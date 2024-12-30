import sys

sys.path.append('../../')
from tools.img import tensor2ndarray, rgb2yuv, yuv2rgb, plot_space_target_space, dct_2d_3c_slide_window, dct_2d_3c_full_scale
from tools.dataset import get_dataloader, get_de_normalization, get_dataset_class_and_scale
# from tools.inject_backdoor import patch_trigger
import numpy as np
import torch
from tqdm import tqdm
from tools.img import fft_2d_3c, ifft_2d_3c
from tools.img import ndarray2tensor
from tools.frft import FRFT
from skimage.metrics import structural_similarity
from skimage.metrics import peak_signal_noise_ratio
import hydra
from omegaconf import DictConfig, OmegaConf
from tools.utils import manual_seed
import random
import matplotlib.pyplot as plt
import os
import argparse
from tools.utils import rm_if_exist
import PIL.Image

def fft_result(args):
    target_folder = args.path
    total = args.total
    path = f'{target_folder}/config.yaml'
    config = OmegaConf.load(path)
    manual_seed(config.seed)
    device = 'cuda:0' 
    num_class, scale = get_dataset_class_and_scale(config.dataset_name)
    train_dl, test_dl = get_dataloader(config.dataset_name, total, config.pin_memory, config.num_workers)

    amp_before = np.zeros((scale, scale, 3), dtype=np.float32)
    amp_after = np.zeros((scale, scale, 3), dtype=np.float32)
    pha_before = np.zeros((scale, scale, 3), dtype=np.float32)
    pha_after = np.zeros((scale, scale, 3), dtype=np.float32)

    batch, labels = next(iter(train_dl))
    batch = batch.to(device=device)

    x_c4show = None
    x_p4show = None

    sys.path.append(target_folder)
    from inject_backdoor import patch_trigger
    for i in tqdm(range(total)):
        x_space = batch[i]  # this is a tensor
        y = labels[i]
        x_space = get_de_normalization(config.dataset_name)(x_space).squeeze()
        x_space_poison = x_space.clone()
        x_space_poison = patch_trigger(x_space_poison, config)  # tensor too
        x_space_poison.clip_(0, 1)
        x_space, x_space_poison = tensor2ndarray(x_space), tensor2ndarray(x_space_poison)
        x_fft = np.fft.fft2(x_space, axes=(0, 1))
        x_p_fft = np.fft.fft2(x_space_poison, axes=(0, 1))
        amp_c, pha_c = np.abs(x_fft), np.angle(x_fft)
        amp_p, pha_p = np.abs(x_p_fft), np.angle(x_p_fft)

        amp_before += amp_c
        pha_before += pha_c
        amp_after += amp_p
        pha_after += pha_p
        if y.item() == 9 and x_p4show is None:
            x_c4show = x_space
            x_p4show = x_space_poison
    amp_before /= total
    amp_after /= total
    pha_before /= total
    pha_after /= total

    # amp_before = amp_before / np.max(amp_before)
    # amp_after = amp_after / np.max(amp_after)
    amp_before = np.log1p(np.abs(amp_before)).astype(np.uint8)
    amp_after = np.log1p(np.abs(amp_after)).astype(np.uint8)

    # pha_before = pha_before / np.max(pha_before)
    # pha_after = pha_after / np.max(pha_after)
    # pha_before = np.log1p(np.abs(pha_before)).astype(np.uint8)
    # pha_after = np.log1p(np.abs(pha_after)).astype(np.uint8)

    amp_before = np.fft.fftshift(amp_before, axes=(0, 1))
    amp_after = np.fft.fftshift(amp_after, axes=(0, 1))

    pha_before = np.fft.fftshift(pha_before, axes=(0, 1))
    pha_after = np.fft.fftshift(pha_after, axes=(0, 1))

    _, ax = plt.subplots(2, 3, figsize=(15, 10))
    for axes in ax.flat:
        axes.set_axis_off()
    ax[0, 0].imshow(x_c4show)
    ax[0, 0].set_title('clean')
    ax[0, 1].imshow(amp_before[:, :, 0])
    ax[0, 1].set_title('clean amp')
    ax[0, 2].imshow(pha_before[:, :, 0])
    ax[0, 2].set_title('clean pha')

    ax[1, 0].imshow(x_p4show)
    ax[1, 0].set_title('poisoned')
    ax[1, 1].imshow(amp_after[:, :, 0])
    ax[1, 1].set_title('poisoned amp')
    ax[1, 2].imshow(pha_after[:, :, 0])
    ax[1, 2].set_title('poisoned pha')
    rm_if_exist(f'{target_folder}/fft_analyze/')
    os.makedirs(f'{target_folder}/fft_analyze', exist_ok=True)
    plt.savefig(f'{target_folder}/fft_analyze/hotmap.png')
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser('')
    parser.add_argument(
        '--path',
        type=str,
        default='/home/chengyiqiu/code/INBA/results/cifar10/inba/20241127121540'
    )
    parser.add_argument(
        '--total',
        type=int,
        default=1024
    )
    args = parser.parse_args()
    fft_result(args)
    
