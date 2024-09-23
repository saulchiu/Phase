import sys
sys.path.append('../')

import torch
import numpy as np
import PIL.Image
from torchvision.transforms.transforms import ToTensor, Resize, Compose

x_c = PIL.Image.open('../data/celeba/img_align_celeba/000001.jpg')
x_p = PIL.Image.open('../data/celeba/img_align_celeba/000002.jpg')
trans = Compose([
    Resize((224, 224)), np.array
])
x_p