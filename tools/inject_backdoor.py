import torch
import torchvision
from torchvision.transforms.transforms import ToTensor, Resize, Compose
from PIL import Image


def patch_trigger(x_0: torch.Tensor, attack_name: str)-> torch.Tensor:
    """
    add a trigger to the original image given attack method
    :param x_0:
    :param attack_name:
    :return: poison image with trigger
    """
    c, h, w = x_0.shape
    trans = Compose([ToTensor(), Resize((h, h))])
    if attack_name == 'blended':
        tg = Image.open('../resource/blended/hello_kitty.jpeg')
        tg = trans(tg)
        x_0 = x_0 * 0.8 + tg * 0.2
    elif attack_name == 'badnet':
        tg = Image.open(f'../resource/badnet/trigger_{h}_3.png')
        mask = Image.open(f'../resource/badnet/mask_{h}_3.png')
        tg = trans(tg)
        mask = trans(mask)
        x_0 = (1 - mask) * x_0 + tg * mask
    return x_0