from torchvision.transforms.transforms import Compose, ToTensor, Resize
from torch.utils.data.dataloader import DataLoader
import torchvision


def get_dataloader(dataset_name: str, batch_size: int) -> (int, Compose, DataLoader):
    """
    given dataset name and batch size, return image scale, transform, dataloader
    :param batch_size:
    :param dataset_name:
    :return: tumple (scale, transform, dataloader)
    """
    scale, trans, dl = None, None, None
    if dataset_name == 'cifar10':
        scale = 32
        trans = Compose([ToTensor(), Resize((scale, scale))])
        ds = torchvision.datasets.CIFAR10(root='../data', train=False, transform=trans)
        dl = DataLoader(dataset=ds, batch_size=batch_size, shuffle=True, num_workers=8)
    elif dataset_name == 'celeba':
        scale = 224
        trans = Compose([ToTensor(), Resize((scale, scale))])
        ds = torchvision.datasets.CelebA(root='../data', split='test', transform=trans)
        dl = DataLoader(dataset=ds, batch_size=batch_size, shuffle=True, num_workers=8)
    elif dataset_name == 'gtsrb':
        scale = 32
        trans = Compose([ToTensor(), Resize((scale, scale))])
        ds = torchvision.datasets.GTSRB(root='../data', split='test', transform=trans)
        dl = DataLoader(dataset=ds, batch_size=batch_size, shuffle=True, num_workers=8)
    return scale, trans, dl