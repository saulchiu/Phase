import sys
sys.path.append('../')
from tools.dataset import get_dataset_class_and_scale, get_dataloader, get_de_normalization
from tools.utils import manual_seed
from tools.img import tensor2ndarray, ndarray2tensor
from tools.inject_backdoor import patch_trigger

import hydra
from omegaconf import DictConfig, OmegaConf
import os
import PIL.Image
import numpy as np

# manual_seed(77)
manual_seed(76)
@hydra.main(version_base=None, config_path='/home/chengyiqiu/code/INBA/config', config_name='default')
def show_phojan(config: DictConfig):
    dl, _ = get_dataloader(config.dataset_name, 8, False, 4)
    batch = None
    for x, y in dl:
        batch = x
        break
    print(batch[0].shape)
    os.makedirs('./phojan_poisoned_images/', exist_ok=True)
    de_norm = get_de_normalization(config.dataset_name)
    for i in range(batch.shape[0]):
        x = de_norm(batch[i]).squeeze()
        x_p = patch_trigger(x, config)
        x_p.clip_(0, 1)
        PIL.Image.Image.save(PIL.Image.fromarray((tensor2ndarray(x)).astype(np.uint8)), f'./phojan_poisoned_images/benign_image_{i}.jpeg')
        PIL.Image.Image.save(PIL.Image.fromarray((tensor2ndarray(x_p)).astype(np.uint8)), f'./phojan_poisoned_images/poisoned_image_{i}.jpeg')




show_phojan()