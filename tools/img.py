import numpy as np
import torch


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
