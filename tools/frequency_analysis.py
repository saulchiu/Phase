import sys

sys.path.append('../')
from tools.img import tensor2ndarray, rgb2yuv, yuv2rgb, plot_space_target_space, dct_2d_3c_slide_window, dct_2d_3c_full_scale
from tools.dataset import get_dataloader
from tools.inject_backdoor import patch_trigger
import numpy as np
import torch
from tqdm import tqdm


if __name__ == '__main__':
    dataset_name = 'imagenette'
    attack = 'duba'
    total = 128
    scale, trans, dl = get_dataloader(dataset_name, total)
    res_before = np.zeros((scale, scale, 3), dtype=np.float32)
    res_after = np.zeros((scale, scale, 3), dtype=np.float32)
    batch: torch.Tensor = next(iter(dl))[0]

    for i in tqdm(range(total)):
        x_space = batch[i]  # this is a tensor
        x_space_poison = patch_trigger(x_space, attack)  # tensor too
        x_space, x_space_poison = tensor2ndarray(x_space), tensor2ndarray(x_space_poison)
        # x_space, x_space_poison = rgb2yuv(x_space), rgb2yuv(x_space_poison)

        x_dct = dct_2d_3c_full_scale(x_space.astype(float))
        x_dct_poison = dct_2d_3c_full_scale(x_space_poison.astype(float))

        res_before += x_dct
        res_after += x_dct_poison
    res_before /= total
    res_after /= total
    x_dct = res_before
    x_dct_poison = res_after
    plot_space_target_space(x_space, x_dct, x_space_poison, x_dct_poison, is_clip=True)
    print((x_dct_poison - x_dct).mean())
