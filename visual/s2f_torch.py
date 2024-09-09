import sys
sys.path.append('../')
from tools.dct import dct_2d
from tools.img import tensor2ndarray

import PIL
import numpy as np
import cv2
import matplotlib.pyplot as plt
import torch
import torchvision
from scipy.fftpack import dct, idct
from PIL import Image
from tqdm import tqdm
from torchvision.transforms.transforms import ToTensor, Resize, Compose
from torch.utils.data.dataloader import DataLoader


def dct2(block):
    return dct(dct(block.T, norm='ortho').T, norm='ortho')


def idct2(block):
    return idct(idct(block.T, norm='ortho').T, norm='ortho')

def clip(data, min=1.5, max=4.5):
    return np.clip(data, min, max)


scale = 224
total = 1000
device = 'cuda:0'
trans = Compose([ToTensor(), Resize((scale, scale))])
tg = Image.open('../resource/blended/hello_kitty.jpeg')
tg = trans(tg)
res_before = torch.zeros_like(tg, device=device)
res_after = torch.zeros_like(tg, device=device)


ds = torchvision.datasets.CelebA(root='../data', split='test', download=False, transform=trans)
dl = DataLoader(dataset=ds, batch_size=total, shuffle=False, num_workers=8)
batch = next(iter(dl))[0]
for i in tqdm(range(batch.shape[0])):
    x_space = batch[i]
    x_space_blended = x_space * 0.8 + tg * 0.2
    x_dct = torch.zeros_like(x_space, device=device)
    for i in range(3):
        # x_dct[:, :, i] = dct2((x_space[:, :, i] * 255).astype(np.uint8))
        x_dct[i, :, :] = dct_2d(x_space[i, :, :])
    x_dct_blended = torch.zeros_like(x_space, device=device)
    for i in range(3):
        # x_dct_blended[:, :, i] = dct2((x_space_blended[:, :, i] * 255).astype(np.uint8))
        x_dct_blended[i, :, :] = dct_2d(x_space_blended[i, :, :])
    res_before += x_dct
    res_after += x_dct_blended

res_before /= total
res_after /= total
x_dct = res_before
x_dct_blended = res_after

x_space = tensor2ndarray(x_space)
x_dct = tensor2ndarray(x_dct)
x_space_blended = tensor2ndarray(x_space_blended)
x_dct_blended = tensor2ndarray(x_dct_blended)

fig, axs = plt.subplots(2, 2, figsize=(15, 8))
axs[0, 0].imshow(x_space)
axs[0, 0].set_title('Original Image')
axs[0, 0].axis('off')

im1 = axs[0, 1].imshow(clip(x_dct[:, :, 0]), cmap='hot')
# im1 = axs[0, 1].imshow(np.log(np.abs(x_dct[:, :, 0]) + 1), cmap='hot')
axs[0, 1].set_title('Original Image DCT (Log Scale)')
axs[0, 1].axis('off')

axs[1, 0].imshow(x_space_blended)
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
