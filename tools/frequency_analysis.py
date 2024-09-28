import sys

sys.path.append('../')
from tools.img import tensor2ndarray, rgb2yuv, yuv2rgb, plot_space_target_space, dct_2d_3c_slide_window, dct_2d_3c_full_scale
from tools.dataset import get_dataloader
from tools.inject_backdoor import patch_trigger
import numpy as np
import torch
from tqdm import tqdm
from tools.img import fft_2d_3c, ifft_2d_3c
from tools.img import ndarray2tensor
from tools.frft import FRFT
from skimage.metrics import structural_similarity
from skimage.metrics import peak_signal_noise_ratio


if __name__ == '__main__':
    dataset_name = 'gtsrb'
    attack = 'inba'
    device = 'cpu'
    visible_tf = 'dct'
    total = 128
    scale, trans, dl = get_dataloader(dataset_name, total)
    res_before = np.zeros((scale, scale, 3), dtype=np.float32)
    res_after = np.zeros((scale, scale, 3), dtype=np.float32)
    batch: torch.Tensor = next(iter(dl))[0]
    batch = batch.to(device=device)
    ssim = 0.
    psnr = 0.
    for i in tqdm(range(total)):
        x_space = batch[i]  # this is a tensor
        x_space_poison = patch_trigger(x_space, attack)  # tensor too
        x_space, x_space_poison = tensor2ndarray(x_space), tensor2ndarray(x_space_poison)
        ssim += structural_similarity(x_space, x_space_poison, win_size=3)
        psnr += peak_signal_noise_ratio(x_space, x_space_poison)
        # x_space, x_space_poison = rgb2yuv(x_space), rgb2yuv(x_space_poison)
        
        if visible_tf == 'dct':
            # DCT 
            x_f = dct_2d_3c_full_scale(x_space.astype(float))
            x_f_poison = dct_2d_3c_full_scale(x_space_poison.astype(float))
        elif visible_tf == 'fft':
            # FFT
            x_f = np.abs(fft_2d_3c(x_space))
            x_f_poison = np.abs(fft_2d_3c(x_space_poison))
        elif visible_tf == 'frft':
            # FRFT
            frft_model = FRFT(3)
            x_space = ndarray2tensor(x_space)
            x_space_poison = ndarray2tensor(x_space_poison)
            x_f = frft_model.FRFT2D(x_space.unsqueeze(0))
            x_f_poison = frft_model.FRFT2D(x_space_poison.unsqueeze(0))
            x_f = x_f.squeeze()
            x_f_poison = x_f_poison.squeeze()
            x_f = tensor2ndarray(x_f)
            x_f_poison = tensor2ndarray(x_f_poison)
            x_space = tensor2ndarray(x_space)
            x_space_poison = tensor2ndarray(x_space_poison)
        res_before += x_f
        res_after += x_f_poison
    res_before /= total
    res_after /= total
    ssim /= total
    psnr /= total
    x_f = res_before
    x_f_poison = res_after
    plot_space_target_space(x_space, x_f, x_space_poison, x_f_poison, is_clip=True)
    print(f'ssim: {ssim:.2f}, psnr: {psnr:.2f}')
