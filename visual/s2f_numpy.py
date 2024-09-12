import sys

sys.path.append('../')
from tools.img import tensor2ndarray, rgb2yuv, yuv2rgb, plot_space_target_space, dct2, idct2
from tools.dataset import get_dataloader
from tools.inject_backdoor import patch_trigger

import PIL
import numpy
import numpy as np
import cv2
import matplotlib.pyplot as plt
import torch
import torchvision

from PIL import Image
from tqdm import tqdm
from torch.utils.data.dataloader import DataLoader
from torchvision.transforms.transforms import ToTensor, Resize, Compose





if __name__ == '__main__':
    dataset_name = 'celeba'
    attack = 'ftrojan'
    total = 1
    window = 224
    scale, trans, dl = get_dataloader(dataset_name, total)
    res_before = np.zeros((scale, scale, 3), dtype=np.float32)
    res_after = np.zeros((scale, scale, 3), dtype=np.float32)
    batch: torch.Tensor = next(iter(dl))[0]

    for i in tqdm(range(total)):
        x_space = batch[i]  # this is a tensor
        x_space_poison = patch_trigger(x_space, attack)  # tensor too
        x_space, x_space_poison = tensor2ndarray(x_space), tensor2ndarray(x_space_poison)
        x_dct = dct2(x_space, window)
        x_dct_blended = dct2(x_space_poison, window)
        res_before += x_dct
        res_after += x_dct_blended
    res_before /= total
    res_after /= total
    x_dct = res_before
    x_dct_blended = res_after
    plot_space_target_space(x_space, x_dct, x_space_poison, x_dct_blended, is_clip=False)
