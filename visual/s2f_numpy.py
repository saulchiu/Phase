import sys

sys.path.append('../')
from tools.img import tensor2ndarray
from tools.dataset import extract_dl
from tools.inject_backdoor import patch_trigger

import PIL
import numpy
import numpy as np
import cv2
import matplotlib.pyplot as plt
import torch
import torchvision
from scipy.fftpack import dct, idct
from PIL import Image
from tqdm import tqdm
from torch.utils.data.dataloader import DataLoader
from torchvision.transforms.transforms import ToTensor, Resize, Compose


def dct2(block):
    return dct(dct(block.T, norm='ortho').T, norm='ortho')


def idct2(block):
    return idct(idct(block.T, norm='ortho').T, norm='ortho')


def clip(data, min=1.5, max=4.5):
    if data.shape[0] == 224:
        return np.clip(data, min, max)
    else:
        return data



dataset_name = 'gtsrb'
attack = 'blended'
total = 1024
scale, trans, dl = extract_dl(dataset_name, total)
res_before = np.zeros((scale, scale, 3), dtype=np.float32)
res_after = np.zeros((scale, scale, 3), dtype=np.float32)
batch = next(iter(dl))[0]

for i in tqdm(range(total)):
    x_space = batch[i] # this is a tensor
    x_space_poison = patch_trigger(x_space, attack) # tensor too
    x_space, x_space_poison = tensor2ndarray(x_space), tensor2ndarray(x_space_poison)
    x_dct = np.zeros_like(x_space, dtype=np.float32)
    for i in range(3):
        x_dct[:, :, i] = dct2((x_space[:, :, i]).astype(np.uint8))
    x_dct_blended = np.zeros_like(x_space_poison, dtype=np.float32)
    for i in range(3):
        x_dct_blended[:, :, i] = dct2((x_space_poison[:, :, i]).astype(np.uint8))
    res_before += x_dct
    res_after += x_dct_blended

res_before /= total
res_after /= total
x_dct = res_before
x_dct_blended = res_after

fig, axs = plt.subplots(2, 2, figsize=(15, 8))
axs[0, 0].imshow(x_space)
axs[0, 0].set_title('Original Image')
axs[0, 0].axis('off')

im1 = axs[0, 1].imshow(clip(x_dct[:, :, 0]), cmap='hot')
# im1 = axs[0, 1].imshow(np.log(np.abs(x_dct[:, :, 0]) + 1), cmap='hot')
axs[0, 1].set_title('Original Image DCT (Log Scale)')
axs[0, 1].axis('off')

axs[1, 0].imshow(x_space_poison)
axs[1, 0].set_title('Blended Image (80% Original, 20% tg)')
axs[1, 0].axis('off')

im2 = axs[1, 1].imshow(clip(x_dct_blended[:, :, 0]), cmap='hot')
# im2 = axs[1, 1].imshow(np.log(np.abs(x_dct_blended[:, :, 0]) + 1), cmap='hot')

axs[1, 1].set_title('Blended Image DCT (Log Scale)')
axs[1, 1].axis('off')

# 添加颜色条
cbar_ax = fig.add_axes([0.92, 0.1, 0.02, 0.35])
fig.colorbar(im2, cax=cbar_ax)

plt.tight_layout()  # 调整布局
plt.show()
