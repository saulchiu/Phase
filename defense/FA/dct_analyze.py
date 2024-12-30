import sys
sys.path.append('../../')
from tools.img import tensor2ndarray, rgb2yuv, yuv2rgb, plot_space_target_space, dct_2d_3c_slide_window, dct_2d_3c_full_scale
from tools.dataset import get_dataloader, get_de_normalization, get_dataset_class_and_scale, get_dataset_normalization, clip_normalized_tensor
# from tools.inject_backdoor import patch_trigger
import numpy as np
import torch
from tqdm import tqdm
from omegaconf import DictConfig, OmegaConf
from tools.utils import manual_seed
import random
import numpy
import argparse
import matplotlib.pyplot as plt
import os
from tools.utils import rm_if_exist
import PIL.Image

def clip(data: numpy.ndarray) -> numpy.ndarray:
    if data.shape[0] > 64:
        return np.clip(a=data, a_min=1.5, a_max=4.5)
    else:
        from scipy.ndimage import gaussian_filter
        data = np.log1p(np.abs(data))
        data = gaussian_filter(data, sigma=2)
        return data
    
def dct_result(args):
    target_folder = args.path
    path = f'{target_folder}/config.yaml'
    config = OmegaConf.load(path)
    manual_seed(config.seed)
    device = 'cuda:0' 
    total = args.total
    is_clip=True

    num_class, scale = get_dataset_class_and_scale(config.dataset_name)
    train_dl, test_dl = get_dataloader(config.dataset_name, total, config.pin_memory, config.num_workers)
    res_before = np.zeros((scale, scale, 3), dtype=np.float32)
    res_after = np.zeros((scale, scale, 3), dtype=np.float32)
    batch, labels = next(iter(train_dl))
    batch = batch.to(device=device)

    x_c4show = None
    x_p4show = None
    de_norm = get_de_normalization(config.dataset_name)
    do_norm = get_dataset_normalization(config.dataset_name)

    sys.path.append(target_folder)
    from inject_backdoor import patch_trigger
    for i in tqdm(range(total)):
        x_space = batch[i]  # this is a tensor
        y = labels[i]
        x_space = de_norm(x_space).squeeze()
        x_space_poison = patch_trigger(x_space, config)  # tensor too
        x_space_poison.clip_(0, 1)
        x_space, x_space_poison = tensor2ndarray(x_space), tensor2ndarray(x_space_poison)
        x_f = dct_2d_3c_full_scale(x_space.astype(float))
        x_f_poison = dct_2d_3c_full_scale(x_space_poison.astype(float))
        res_before += x_f
        res_after += x_f_poison
        if y.item() == 9 and x_p4show is None:
            x_c4show = x_space
            x_p4show = x_space_poison
    res_before /= total
    res_after /= total
    # plot_space_target_space(x_c4show, x_f, x_p4show, x_f_poison, is_clip=True)
    if is_clip:
        x_target = clip(res_before)
        x_process_target = clip(res_after)
    fig, axs = plt.subplots(2, 2, figsize=(15, 10))
    axs[0, 0].imshow(x_c4show)
    axs[0, 0].set_title(f'Original Image')
    axs[0, 0].axis('off')

    axs[0, 1].imshow(x_target[:, :, 0], cmap='hot')
    axs[0, 1].set_title('Original Image DCT')
    axs[0, 1].axis('off')

    axs[1, 0].imshow(x_p4show)
    axs[1, 0].set_title(f'after (attack) process')
    axs[1, 0].axis('off')

    im2 = axs[1, 1].imshow(x_process_target[:, :, 0], cmap='hot')
    axs[1, 1].set_title(f'(attacked) img in target space')
    axs[1, 1].axis('off')
    cbar_ax = fig.add_axes([0.92, 0.1, 0.02, 0.35])
    fig.colorbar(im2, cax=cbar_ax)
    plt.tight_layout()
    rm_if_exist(f'{target_folder}/dct_analyze/')
    os.makedirs(f'{target_folder}/dct_analyze', exist_ok=True)
    plt.savefig(f'{target_folder}/dct_analyze/hotmap.png')
    plt.show()


    

if __name__ == '__main__':
    parser = argparse.ArgumentParser('')
    parser.add_argument(
        '--path',
        type=str,
        default='/home/chengyiqiu/code/INBA/results/cifar10/inba/convnext/20241128150334'
    )
    parser.add_argument(
        '--total',
        type=int,
        default=1024
    )
    args = parser.parse_args()
    dct_result(args)

