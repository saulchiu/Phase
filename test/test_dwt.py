import sys

sys.path.append('../')
from tools.img import dwt_3c, idwt_3c, dct_2d_full_scale
import pywt
import PIL.Image
import numpy as np
from matplotlib import pyplot as plt
from torchvision.transforms.transforms import Compose, ToTensor, Resize

wavelet = 'haar'
titles = ['Approximation', ' Horizontal detail',
          'Vertical detail', 'Diagonal detail', 'img_reconstructed', 'dct']
img = PIL.Image.open('../resource/example/000001.jpg')
trans = Compose([Resize((224, 224)), np.array])
img = trans(img) / 255.
LL_r, (LH_r, HL_r, HH_r) = pywt.dwt2(img[:, :, 0], wavelet)
LL_g, (LH_g, HL_g, HH_g) = pywt.dwt2(img[:, :, 1], wavelet)
LL_b, (LH_b, HL_b, HH_b) = pywt.dwt2(img[:, :, 2], wavelet)

LL = np.zeros(shape=(LL_r.shape[0], LL_r.shape[1], 3))
LH = np.zeros(shape=(LH_r.shape[0], LH_r.shape[1], 3))
HL = np.zeros(shape=(HL_r.shape[0], HL_r.shape[1], 3))
HH = np.zeros(shape=(HH_r.shape[0], HH_r.shape[1], 3))

LL[:, :, 0], LL[:, :, 1], LL[:, :, 2] = LL_r, LL_g, LL_b
LH[:, :, 0], LH[:, :, 1], LH[:, :, 2] = LH_r, LH_g, LH_b
HL[:, :, 0], HL[:, :, 1], HL[:, :, 2] = HL_r, HL_g, HL_b
HH[:, :, 0], HH[:, :, 1], HH[:, :, 2] = HH_r, HH_g, HH_b

# inject trigger
window = 32
pos_list = [(15, 15), (32, 32)]
magnitude = 0.32

LL = np.transpose(LL, (2, 0, 1))
ch, w, h = LL.shape
for i in range(ch):
    for j in range(0, w, window):
        for k in range(0, h, window):
            for pos in pos_list:
                LL[i][min(j + pos[0], w - 1), min(k + pos[1], h - 1)] += magnitude
LL = np.transpose(LL, (1, 2, 0))

img_reconstructed = idwt_3c([LL, (LH, HL, HH)], wavelet)

img_dct = dct_2d_full_scale(img_reconstructed * 255.)
img_dct = np.clip(img_dct, 1.5, 4.5)

# clip
LL = np.clip(LL, 0., 1.)
LH = np.clip(LH, 0., 1.)
LH = np.clip(LH, 0., 1.)
HH = np.clip(HH, 0., 1.)

fig = plt.figure(figsize=(12, 3))
for i, a in enumerate([LL, LH, HL, HH, img_reconstructed, img_dct]):
    ax = fig.add_subplot(1, 6, i + 1)
    ax.imshow(a, cmap='hot')
    ax.set_title(titles[i], fontsize=10)
    ax.set_xticks([])
    ax.set_yticks([])

fig.tight_layout()
plt.show()
