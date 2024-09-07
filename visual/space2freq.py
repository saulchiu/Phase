import PIL
import numpy as np
import cv2
import matplotlib.pyplot as plt
import torch
import torchvision
from scipy.fftpack import dct, idct
from PIL import Image
from tqdm import tqdm


def dct2(block):
    return dct(dct(block.T, norm='ortho').T, norm='ortho')


def idct2(block):
    return idct(idct(block.T, norm='ortho').T, norm='ortho')

def clip(data, min=1.5, max=4.5):
    return np.clip(data, min, max)


scale = 224
total = 100
tg = cv2.imread('../resource/blended/hello_kitty.jpeg')
tg = cv2.cvtColor(tg, cv2.COLOR_BGR2RGB)
tg = cv2.resize(tg, (scale, scale))
res_before = np.zeros_like(tg, dtype=np.float32)
res_after = np.zeros_like(tg, dtype=np.float32)

for i in tqdm(range(total)):
    ds = torchvision.datasets.CelebA(root='../data', split='train', download=False)
    x_space = np.array(ds[i][0])
    # x_space = cv2.cvtColor(x_space, cv2.COLOR_BGR2RGB)
    x_space = cv2.resize(x_space, (scale, scale))
    x_space_blended = (x_space * 0.8 + tg * 0.2).astype(np.uint8)
    x_dct = np.zeros_like(x_space, dtype=np.float32)
    for i in range(3):
        x_dct[:, :, i] = dct2((x_space[:, :, i] * 255).astype(np.uint8))
    x_dct_blended = np.zeros_like(x_space_blended, dtype=np.float32)
    for i in range(3):
        x_dct_blended[:, :, i] = dct2((x_space_blended[:, :, i] * 255).astype(np.uint8))
    res_before += x_dct
    res_after += x_dct_blended
res_before /= total
res_after /= total
x_dct = res_before
x_dct_blended = res_after
# 绘制原图像与DCT的结果
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
