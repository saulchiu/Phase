import sys
sys.path.append('../')
from tools.utils import manual_seed
from omegaconf import OmegaConf
import torch
from tools.dataset import get_dataloader
import torch.nn.functional as F
from torchmetrics.image import StructuralSimilarityIndexMeasure
from torchmetrics.image import PeakSignalNoiseRatio, LearnedPerceptualImagePatchSimilarity
from skimage.metrics import peak_signal_noise_ratio
# from tools.inject_backdoor import patch_trigger
from tools.dataset import get_de_normalization
from tools.img import tensor2ndarray
import tqdm
import matplotlib.pyplot as plt


def cal_ssim_psnr(target_folder):
    # this script is indepedent on model, except INBA
    # target_folder = '../' + target_folder
    path = f'{target_folder}/config.yaml'
    config = OmegaConf.load(path)
    config.attack.mode = "eval"
    manual_seed(config.seed)
    device = f'cuda:{config.device}'
    train_dl, test_dl = get_dataloader(config.dataset_name, config.batch, config.pin_memory, config.num_workers)
    lp_function = LearnedPerceptualImagePatchSimilarity(net_type='squeeze').to(f'cuda:{config.device}')
    ssim_function = StructuralSimilarityIndexMeasure(data_range=1.0).to(f'cuda:{config.device}')
    psnr_function = PeakSignalNoiseRatio(data_range=1.0).to(f'cuda:{config.device}')
    de_norm = get_de_normalization(config.dataset_name)

    sys.path.append('./run')
    sys.path.append(target_folder)
    from inject_backdoor import patch_trigger

    ssim_metric = 0.
    psnr_metric = 0.
    lpips_metric = 0.
    total = 0

    for batch, _ in train_dl:
        batch = de_norm(batch)
        poison_batch = []
        for i in range(batch.shape[0]):
            x_c = batch[i]
            x_p = patch_trigger(x_c.squeeze(), config)
            poison_batch.append(x_p)
        poison_batch = torch.stack(poison_batch, dim=0)
        batch = batch.to(f'cuda:{config.device}')
        poison_batch = poison_batch.to(f'cuda:{config.device}')
        poison_batch.clip_(0, 1)
        batch.clip_(0, 1)
        ssim_metric += ssim_function(poison_batch, batch).item()
        psnr_metric += psnr_function(poison_batch, batch).item()
        lpips_metric += lp_function(poison_batch, batch).item()
        total += 1

    for batch, _ in test_dl:
        batch = de_norm(batch)
        poison_batch = []
        for i in range(batch.shape[0]):
            x_c = batch[i]
            x_p = patch_trigger(x_c.squeeze(), config)
            poison_batch.append(x_p)
        poison_batch = torch.stack(poison_batch, dim=0)
        poison_batch.clip_(0, 1)
        batch = batch.to(f'cuda:{config.device}')
        poison_batch = poison_batch.to(f'cuda:{config.device}')
        ssim_metric += ssim_function(poison_batch, batch).item()
        psnr_metric += psnr_function(poison_batch, batch).item()
        lpips_metric += lp_function(poison_batch, batch).item()
        total += 1


    ssim_metric /= total
    psnr_metric /= total
    lpips_metric /= total

    print(f'total image: {total * config.batch}\n SSIM: {ssim_metric:.8f}\n PSNR: {psnr_metric: .8f}\n LPIPS: {lpips_metric: .8f}')

    _, ax = plt.subplots(1, 2)
    ax[0].imshow(tensor2ndarray(x_c))
    ax[0].set_title('x_c')
    ax[1].imshow(tensor2ndarray(x_p))
    ax[1].set_title('x_p')
    plt.show()
    return ssim_metric, psnr_metric, lpips_metric


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Process some paths.')
    parser.add_argument('--path', type=str, help='The path to the target folder.')
    args = parser.parse_args()
    target_folder = args.path
    cal_ssim_psnr(target_folder=target_folder)