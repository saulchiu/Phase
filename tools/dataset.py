from torchvision.transforms.transforms import Compose, ToTensor, Resize, Normalize
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
        dl = DataLoader(dataset=ds, batch_size=batch_size, shuffle=True, num_workers=0)
    elif dataset_name == 'celeba':
        scale = 224
        trans = Compose([ToTensor(), Resize((scale, scale))])
        ds = torchvision.datasets.CelebA(root='../data', split='test', transform=trans)
        dl = DataLoader(dataset=ds, batch_size=batch_size, shuffle=True, num_workers=0)
    elif dataset_name == 'gtsrb':
        scale = 32
        trans = Compose([ToTensor(), Resize((scale, scale))])
        ds = torchvision.datasets.GTSRB(root='../data', split='test', transform=trans)
        dl = DataLoader(dataset=ds, batch_size=batch_size, shuffle=True, num_workers=0)
    elif dataset_name == 'imagenette':
        scale = 224
        trans = Compose([ToTensor(), Resize((scale, scale))])
        ds = torchvision.datasets.Imagenette(root='../data', split='train', transform=trans)
        dl = DataLoader(dataset=ds, batch_size=batch_size, shuffle=True, num_workers=0)
    elif dataset_name == 'pubfig':
        pass
    return scale, trans, dl

import torch
from torch.utils.data import Dataset, DataLoader

class List2Dataset(Dataset):
    def __init__(self, data_list):
        self.data_list = data_list

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        x, y = self.data_list[idx]
        return x, y


def get_dataset_normalization(dataset_name):
    # this function is from BackdoorBench
    # given name, return the default normalization of images in the dataset
    if dataset_name == "cifar10":
        # from wanet
        dataset_normalization = Normalize([0.4914, 0.4822, 0.4465], [0.247, 0.243, 0.261])
    elif dataset_name == "gtsrb" or dataset_name == "celeba":
        dataset_normalization = Normalize([0, 0, 0], [1, 1, 1])
    elif dataset_name == 'cifar100':
        dataset_normalization = Normalize([0.5071, 0.4865, 0.4409], [0.2673, 0.2564, 0.2762])
    elif dataset_name == "mnist":
        dataset_normalization = Normalize([0.5], [0.5])
    elif dataset_name == 'tiny':
        dataset_normalization = Normalize([0.4802, 0.4481, 0.3975], [0.2302, 0.2265, 0.2262])
    elif dataset_name == 'imagenet':
        dataset_normalization = Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    elif dataset_name == 'imagenette':
        # mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]
        dataset_normalization = Normalize(mean=[0.4648, 0.4543, 0.4247], std=[0.2785, 0.2735, 0.2944])
    else:
        raise NotImplementedError(dataset_name)
    return dataset_normalization