import torch
import torch.nn as nn

def CLP(net, u):
    params = net.state_dict()
    for name, m in net.named_modules():
        if isinstance(m, nn.Conv2d):
            channel_lips = []
            for idx in range(m.weight.shape[0]):
                weight = m.weight[idx]
                weight = weight.reshape(weight.shape[0], -1).cpu()
                channel_lips.append(torch.svd(weight)[1].max())
                
            channel_lips = torch.Tensor(channel_lips)

            index = torch.where(channel_lips>channel_lips.mean() + u * channel_lips.std())[0]

            params[name+'.weight'][index] = 0
            print(index)
        
        elif isinstance(m, nn.BatchNorm2d):
            std = m.running_var.sqrt()
            weight = m.weight
            channel_lips = (weight / std).abs()

            index = torch.where(channel_lips>channel_lips.mean() + u * channel_lips.std())[0]

            params[name+'.weight'][index] = 0
            params[name+'.bias'][index] = 0
            print(index)


    net.load_state_dict(params)
    return net


import torch
import torch.nn.functional as F

import os
import argparse
from tqdm import tqdm


def val(net, data_loader):
    with torch.no_grad():
        net.eval()
        n_correct = 0
        n_total = 0

        for images, targets in data_loader:
            images, targets = images.to(args.device), targets.to(args.device)

            logits = net(images)
            prediction = logits.argmax(-1)

            n_correct += (prediction==targets).sum()
            n_total += targets.shape[0]
            
        acc = n_correct / n_total * 100

    return acc


def main(args):
    import sys
    sys.path.append('../')
    from tools.utils import manual_seed
    from tools.dataset import get_dataset_class_and_scale, get_train_and_test_dataset, get_dataloader, PoisonDataset
    from omegaconf import OmegaConf
    from torch.utils.data.dataloader import DataLoader

    target_folder = args.path
    path = f'{target_folder}/config.yaml'
    config = OmegaConf.load(path)
    manual_seed(config.seed)
    device = f'cuda:{config.device}'
    num_class, _ = get_dataset_class_and_scale(config.dataset_name)
    if config.model == "resnet18":
        from models.preact_resnet import PreActResNet18
        net = PreActResNet18(num_class)
    elif config.model == "repvgg":
        from repvgg_pytorch.repvgg import RepVGG
        net = RepVGG(num_blocks=[2, 4, 14, 1], num_classes=num_class, width_multiplier=[1.5, 1.5, 1.5, 2.75]).to(device=f'cuda:{config.device}')
    else:
        raise NotImplementedError
    ld = torch.load(f'{target_folder}/results.pth', map_location=device)
    net.load_state_dict(ld['model'])
    train_ds, test_ds = get_train_and_test_dataset(config.dataset_name)
    train_dl, test_dl = get_dataloader(config.dataset_name, config.batch, config.pin_memory, config.num_workers)
    num_class, scale = get_dataset_class_and_scale(config.dataset_name)
    net.to(device)

    bd_config = config.copy()
    bd_config.ratio = 1
    bd_train_ds = PoisonDataset(train_ds, config)
    bd_test_ds = PoisonDataset(test_ds, bd_config)

    train_loader = train_dl
    val_loader = test_dl
    test_clean_loader = test_dl
    test_poisoned_loader = DataLoader(dataset=bd_test_ds, batch_size=config.batch, shuffle=False, num_workers=11)
    print('Before prunning')
    acc = val(net, train_loader)
    print('Training accuracy: %.2f' % acc)
    acc = val(net, val_loader)
    print('Validation accuracy: %.2f' % acc)
    acc, asr = val(net, test_clean_loader), val(net, test_poisoned_loader)
    print('Test clean accuracy: %.2f' % acc)
    print('Test attack success rate: %.2f' % asr)
    
    CLP(net, args.u)
    print('After CLP prunning')
    acc = val(net, train_loader)
    print('Training accuracy: %.2f' % acc)
    acc = val(net, val_loader)
    print('Validation accuracy: %.2f' % acc)
    acc, asr = val(net, test_clean_loader), val(net, test_poisoned_loader)
    print('Test clean accuracy: %.2f' % acc)
    print('Test attack success rate: %.2f' % asr)

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='PyTorch Backdoor Training') 

    parser.add_argument("--path", default="../results/cifar10/badnet/20241006002653_resnet18", type=str)
    
    parser.add_argument('--model', default='resnet18', type=str,
                        help='network structure choice')
    parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')

    # Pruning options
    parser.add_argument('--batch-size', default=500, type=int, metavar='N',
                        help='batch size.')
    parser.add_argument('-u', default=2.8, type=float,
                        help='threshold hyperparameter')
    # Checkpoints
    parser.add_argument('-c', '--checkpoint', default='./ckpt', type=str, metavar='PATH',
                        help='path to save checkpoint (default: checkpoint)')

    # Miscs
    parser.add_argument('--manual-seed', default=0, type=int, help='manual seed')

    # Device options
    parser.add_argument('--device', default='cuda:0', type=str,
                        help='device used for training')

    # data path
    parser.add_argument('--dataset', type=str, default='cifar10')
    parser.add_argument('--dataset-dir', type=str, default='./dataset')

    # backdoor setting
    parser.add_argument('--attack-type', type=str, default='badnets')
    parser.add_argument('--target_label', type=int, default=0, help='backdoor target label.')
    parser.add_argument('--poisoning-rate', type=float, default=0.1, help='backdoor training sample ratio.')
    parser.add_argument('--trigger-size', type=int, default=3, help='size of square backdoor trigger.')
    args = parser.parse_args()

    torch.manual_seed(args.manual_seed)
    torch.cuda.manual_seed(args.manual_seed)
    torch.backends.cudnn.deterministic=True

    main(args)
