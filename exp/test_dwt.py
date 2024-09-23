import sys

sys.path.append('../')
from tools.img import dwt_2d_3c, idwt_2d_3c, dct_2d_3c_full_scale
import pywt
import PIL.Image
import numpy as np
from matplotlib import pyplot as plt
from torchvision.transforms.transforms import Compose, ToTensor, Resize

wavelet = 'haar'
titles = ['Approximation', ' Horizontal detail',
          'Vertical detail', 'Diagonal detail', 'img_reconstructed', 'dct']
img = PIL.Image.open('../data/celeba/img_align_celeba/000001.jpg')
trans = Compose([Resize((224, 224)), np.array])
img = trans(img) / 255.
coeff = pywt.wavedec2(img, wavelet, level=3)
[cl,(cH3,cV3,cD3),(cH2,cV2,cD2),(cH1,cV1,cD1)] = coeff



img_reconstructed = pywt.idwt2(coeffs=(LL, (LH, HL, HH)), wavelet=wavelet)

fig = plt.figure(figsize=(12, 3))
for i, a in enumerate([LL, LH, HL, HH, img_reconstructed]):
    ax = fig.add_subplot(1, 5, i + 1)
    ax.imshow(a)
    ax.set_title(titles[i], fontsize=10)
    ax.set_xticks([])
    ax.set_yticks([])

fig.tight_layout()
plt.show()
print((img - img_reconstructed).sum())
