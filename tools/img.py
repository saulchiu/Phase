import cv2
import numpy
import numpy as np
import torch
from matplotlib import pyplot as plt
import cv2
from scipy.fftpack import dct, idct


def ndarray2tensor(nd: np.ndarray) -> torch.Tensor:
    """
    Convert a numpy array to a torch tensor.
    :param nd: a numpy array, e.g., shape (32, 32, 3)
    :return: a torch tensor, e.g., with shape (3, 32, 32)
    """
    # Swap the axes so that channels are first, from (H, W, C) to (C, H, W)
    tensor = (torch.from_numpy(nd) / 255.).permute(2, 0, 1)
    return tensor


def tensor2ndarray(t: torch.Tensor) -> np.ndarray:
    """
    Convert a torch tensor to a numpy array.
    :param t: a tensor, e.g., with shape (3, 32, 32)
    :return: a numpy array, e.g., with shape (32, 32, 3)
    """
    # Permute the tensor to swap back to (H, W, C) from (C, H, W)
    nd = (t * 255.).cpu().permute(1, 2, 0).numpy().astype(np.uint8)
    return nd


def dct2(x_0: numpy.ndarray, window: int = 4) -> numpy.ndarray:
    """
    :param window: a shifted window
    :param x_0: space domain, ndarray, wanted shape (w, h, ch)
    :return: frequency domain (after DCT), ndarray
    """
    x_0 = x_0.copy()
    x_0 = np.transpose(x_0, axes=(2, 0, 1))
    for ch in range(x_0.shape[0]):
        for w in range(0, x_0.shape[1], window):
            for h in range(0, x_0.shape[2], window):
                sub_dct = cv2.dct(x_0[ch][w:w + window, h:h + window].astype(np.float32))
                x_0[ch][w:w + window, h:h + window] = sub_dct
    x_0 = np.transpose(x_0, axes=(1, 2, 0))
    return x_0


def idct2(x_0: numpy.ndarray, window: int = 4) -> numpy.ndarray:
    """
    :param window: a shifted window
    :param x_0: frequency domain (after DCT), ndarray
    :return: string domain, ndarray
    """
    x_0 = x_0.copy()
    x_0 = np.transpose(x_0, axes=(2, 0, 1))
    for ch in range(x_0.shape[0]):
        for w in range(0, x_0.shape[1], window):
            for h in range(0, x_0.shape[2], window):
                sub_dct = cv2.idct(x_0[ch][w:w + window, h:h + window].astype(np.float32))
                x_0[ch][w:w + window, h:h + window] = sub_dct
    x_0 = np.transpose(x_0, axes=(1, 2, 0))
    return x_0

def fft2(block: numpy.ndarray) -> numpy.ndarray:
    return numpy.fft.fft2(block)


def ifft2(block: numpy.ndarray) -> numpy.ndarray:
    return numpy.fft.ifft2(block)


def rgb2yuv(x_rgb: numpy.ndarray) -> numpy.ndarray:
    """
    this function converts RGB image to YUV image both a single image and a batch
    :param x_rgb:
    :return:
    """
    if len(x_rgb.shape) == 4:  # batch
        x_yuv = np.zeros(x_rgb.shape, dtype=float)
        for i in range(x_rgb.shape[0]):
            img = cv2.cvtColor(x_rgb[i].astype(np.uint8), cv2.COLOR_RGB2YUV)
            x_yuv[i] = img
        return x_yuv
    return cv2.cvtColor(x_rgb.astype(np.uint8), cv2.COLOR_RGB2YUV)


def yuv2rgb(x_yuv):
    """
    this function converts YUV image to RGB image both a single image and a batch
    :param x_yuv:
    :return:
    """
    if len(x_yuv.shape) == 4:  # batch
        x_rgb = np.zeros(x_yuv.shape, dtype=float)
        for i in range(x_yuv.shape[0]):
            img = cv2.cvtColor(x_yuv[i].astype(np.uint8), cv2.COLOR_YUV2RGB)
            x_rgb[i] = img
        return x_rgb
    return cv2.cvtColor(x_yuv.astype(np.uint8), cv2.COLOR_YUV2RGB)


def clip(data: numpy.ndarray, min=1.5, max=4.5) -> numpy.ndarray:
    return np.clip(data, min, max)


def plot_space_target_space(x_space: numpy.ndarray, x_target, x_process_space, x_process_target, is_clip: bool = False):
    """
    compare the image in original domain (e.g., space domain) and target domain (e.g., frequency domain)
    :param x_space:
    :param x_target: the image in target domain after target transform
    :param x_process_space:
    :param x_process_target: the processed image (e.g., poisoning image) in target domain after target transform
    :param is_clip: choose whether clip the value in target domain
    :return:
    """
    if is_clip:
        x_target = clip(x_target)
        x_process_target = clip(x_process_target)
    fig, axs = plt.subplots(2, 2, figsize=(15, 8))
    axs[0, 0].imshow(x_space)
    axs[0, 0].set_title('Original Image')
    axs[0, 0].axis('off')
    im1 = axs[0, 1].imshow(clip(x_target[:, :, 0]), cmap='hot')
    axs[0, 1].set_title('Original Image DCT')
    axs[0, 1].axis('off')
    axs[1, 0].imshow(x_process_space)
    axs[1, 0].set_title(f'after (attack) process')
    axs[1, 0].axis('off')
    im2 = axs[1, 1].imshow(clip(x_process_target[:, :, 0]), cmap='hot')
    axs[1, 1].set_title(f'(attacked) img in target space')
    axs[1, 1].axis('off')
    cbar_ax = fig.add_axes([0.92, 0.1, 0.02, 0.35])
    fig.colorbar(im2, cax=cbar_ax)
    plt.tight_layout()
    plt.show()
