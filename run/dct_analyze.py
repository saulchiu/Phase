import sys

sys.path.append('../')
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
from models.preact_resnet import PreActResNet18
from tools.utils import manual_seed
import random
import numpy

if __name__ == '__main__':
    target_folder = '../' + 'results/imagenette/duba/20241021012349'
    path = f'{target_folder}/config.yaml'
    config = OmegaConf.load(path)
    manual_seed(config.seed)
    device = 'cpu' 
    visible_tf = 'dct'
    total = 1024
    num_class, scale = get_dataset_class_and_scale(config.dataset_name)
    train_dl, test_dl = get_dataloader(config.dataset_name, total, config.pin_memory, config.num_workers)
    res_before = np.zeros((scale, scale, 3), dtype=np.float32)
    res_after = np.zeros((scale, scale, 3), dtype=np.float32)
    batch, labels = next(iter(train_dl))
    batch = batch.to(device=device)

    x_c4show = None
    x_p4show = None
    sys.path.append('./run')
    sys.path.append(target_folder)
    from inject_backdoor import patch_trigger
    for i in tqdm(range(total)):
        x_space = batch[i]  # this is a tensor
        y = labels[i]
        x_space = get_de_normalization(config.dataset_name)(x_space).squeeze()
        x_space_poison = patch_trigger(x_space, config)  # tensor too
        x_space_poison = torch.clip(x_space_poison, 0, 1)
        x_space, x_space_poison = tensor2ndarray(x_space), tensor2ndarray(x_space_poison)
        x_f = dct_2d_3c_full_scale(x_space.astype(float))
        x_f_poison = dct_2d_3c_full_scale(x_space_poison.astype(float))
        res_before += x_f
        res_after += x_f_poison
        # if y.item() == 9 and x_p4show is None:
        if y.item() == 1:
            x_c4show = x_space
            x_p4show = x_space_poison
    res_before /= total
    res_after /= total
    # x_f = np.fft.fftshift(res_before, axes=(0, 1))
    # x_f_poison = np.fft.fftshift(res_after, axes=(0, 1))

    plot_space_target_space(x_c4show, x_f, x_p4show, x_f_poison, is_clip=True)
