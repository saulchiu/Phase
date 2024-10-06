import sys

sys.path.append('../')
from tools.img import tensor2ndarray, rgb2yuv, yuv2rgb, plot_space_target_space, dct_2d_3c_slide_window, dct_2d_3c_full_scale
from tools.dataset import get_dataloader, get_de_normalization, get_dataset_class_and_scale
from tools.inject_backdoor import patch_trigger
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


if __name__ == '__main__':
    target_folder = '../' + 'results/cifar10/inba/20241006063029_wind32'
    path = f'{target_folder}/config.yaml'
    config = OmegaConf.load(path)
    manual_seed(config.seed)
    device = 'cpu' 
    visible_tf = 'dct'
    total = 1024
    _, scale = get_dataset_class_and_scale(config.dataset_name)
    _, dl = get_dataloader(config.dataset_name, total, config.pin_memory, config.num_workers)
    res_before = np.zeros((scale, scale, 3), dtype=np.float32)
    res_after = np.zeros((scale, scale, 3), dtype=np.float32)
    batch: torch.Tensor = next(iter(dl))[0]
    batch = batch.to(device=device)
    ssim = 0.
    psnr = 0.
    lpip = 0.
    import lpips
    loss_fn_alex = lpips.LPIPS(net='alex') # best forward scores
    # loss_fn_vgg = lpips.LPIPS(net='vgg') # closer to "traditional" perceptual loss, when used for optimization

    # load model
    if config.model == "resnet18":
        net = PreActResNet18()
        ld = torch.load(f'{target_folder}/results.pth', map_location=device)
        net.load_state_dict(ld['model'])
        net.eval()
    else:
        raise NotImplementedError
    
    for i in tqdm(range(total)):
        x_space = batch[i]  # this is a tensor
        x_space_poison = patch_trigger(
            get_de_normalization(config.dataset_name)(x_space).squeeze()
            , config)  # tensor too
        if i == total - 1:
            y_clean = net(x_space.unsqueeze(0))
            y_poison = net(x_space_poison.unsqueeze(0))
            _, predicted_clean = torch.max(y_clean, -1)
            _, predicted_poison = torch.max(y_poison, -1)
        x_space = get_de_normalization(config.dataset_name)(x_space).squeeze()
        lpip += loss_fn_vgg(x_space, x_space_poison)
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
    lpip /= total
    x_f = res_before
    x_f_poison = res_after

    plot_space_target_space(x_space, predicted_clean.item(), x_f, x_space_poison, predicted_poison.item(), x_f_poison, is_clip=False)
    print(f'ssim: {ssim:.3f}, psnr: {psnr:.2f}, lpips: {lpip.item(): .5f}')
    # print(predicted_clean, predicted_poison)
