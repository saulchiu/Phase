import sys, os
import math
sys.path.append('../')

import argparse
from pprint import  pformat
import numpy as np
import torch
import logging
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader

from tools.utils import manual_seed
from omegaconf import OmegaConf, DictConfig
from tools.dataset import get_dataset_class_and_scale, get_dataloader, get_train_and_test_dataset, PoisonDataset

import torch
import torch.nn as nn
def test(model, test_data, device,multi=False):

    model.eval()
    
    metrics = {
        'test_correct': 0,
        'test_loss': 0,
        'test_total': 0
    }
    criterion = nn.CrossEntropyLoss()
    tot_tar_list = []
    cor_tar_list = []
    with torch.no_grad():
        for batch_idx, (x, target, *additional_info) in enumerate(test_data):
            x = x.to(device)
            target = target.to(device)
            if multi:
                pred,_ = model(x)
            else:
                pred = model(x)
            loss = criterion(pred, target.long())

            _, predicted = torch.max(pred, -1)
            correct_mask = predicted.eq(target)
            for cor, tar in zip(correct_mask,target):
                tot_tar_list.append(int(tar.item()))
                if cor:
                    cor_tar_list.append(int(tar.item()))
            correct = correct_mask.sum()
            metrics['test_correct'] += correct.item()
            metrics['test_loss'] += loss.item() * target.size(0)
            metrics['test_total'] += target.size(0)
            
    return metrics

class LabelSmoothingLoss(nn.Module):
    def __init__(self, classes=10, smoothing=0.1, dim=-1):
        super(LabelSmoothingLoss, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.cls = classes
        self.dim = dim

    def forward(self, pred, target):
        pred = pred.log_softmax(dim=self.dim)
        true_dist = torch.zeros_like(pred)
        true_dist.fill_(self.smoothing / (self.cls - 1))
        true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        return torch.mean(torch.sum(-true_dist * pred, dim=self.dim))


def add_args(parser):
    parser.add_argument('--device', type = str)
    parser.add_argument('--ft_mode', type = str, default='fst')
    
    parser.add_argument('--attack', type = str, )
    parser.add_argument('--attack_label_trans', type=str, default='all2one',
                        help='which type of label modification in backdoor attack'
                        )
    parser.add_argument('--pratio', type=float, default=0.1,
                        help='the poison rate '
                        )
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--dataset', type=str,
                        help='which dataset to use'
                        )
    parser.add_argument('--path', type=str)
    parser.add_argument('--attack_target', type=int,default=0,
                        help='target class in all2one attack')
    parser.add_argument('--batch_size', type=int,default=128)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--random_seed', default=0,type=int,
                        help='random_seed')
    parser.add_argument('--model', type=str,
                        help='choose which kind of model')
    
    parser.add_argument('--split_ratio', type=float, default='0.02',
                        help='part of the training set for defense')
    
    parser.add_argument('--log', action='store_true', default=True,
                        help='record the log')
    parser.add_argument('--pre', action='store_true', help='load pre-trained weights')
    parser.add_argument('--save', action='store_true', help='save the model checkpoint')
    parser.add_argument('--linear_name', type=str, default='linear', help='name for the linear classifier')
    parser.add_argument('--lb_smooth', type=float, default=None, help='label smoothing')
    parser.add_argument('--alpha', type=float, default=0.1, help='fst')
    return parser

def main():
    parser = (add_args(argparse.ArgumentParser(description=sys.argv[0])))
    args = parser.parse_args()

    target_folder = args.path
    path = f'{target_folder}/config.yaml'
    config = OmegaConf.load(path)
    manual_seed(config.seed)
    device = f'cuda:{config.device}'
    num_class, _ = get_dataset_class_and_scale(config.dataset_name)
    if config.model == "resnet18":
        from classifier_models.preact_resnet import PreActResNet18
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

    ### 1. config args, save_path, fix random seed
    

    args.num_classes = num_class
    args.input_height, args.input_width, args.input_channel = scale, scale, 3
    args.img_size = (args.input_height, args.input_width, args.input_channel)
    # args.dataset_path = f"{args.dataset_path}/{args.dataset}"
    

    
    
    if args.lb_smooth is not None:
        lbs_criterion = LabelSmoothingLoss(classes=num_class, smoothing=args.lb_smooth)
    device = f'cuda:{config.device}' if device != 'cpu' else device
    print(args.ft_mode)
    if args.ft_mode == 'fe-tuning':
        init = True
        log_name = 'FE-tuning'
    elif args.ft_mode == 'ft-init':
        init = True
        log_name = 'FT-init'
    elif args.ft_mode == 'ft':
        init = False
        log_name = 'FT'
    elif args.ft_mode == 'lp':
        init = False
        log_name = 'LP'
    elif args.ft_mode == 'fst':
        assert args.alpha is not None
        init = True
        log_name = 'FST'
    else:
        raise NotImplementedError('Not implemented method.')


        
    args.folder_path = target_folder
    args.save_path = f'{target_folder}/{args.ft_mode}'
    os.makedirs(args.save_path, exist_ok=True)


    logFormatter = logging.Formatter(
        fmt='%(asctime)s [%(levelname)-8s] [%(filename)s:%(lineno)d] %(message)s',
        datefmt='%Y-%m-%d:%H:%M:%S',
    )
    logger = logging.getLogger()

    if args.log:
        fileHandler = logging.FileHandler(f'{args.save_path}/ft.log')
        fileHandler.setFormatter(logFormatter)
        logger.addHandler(fileHandler)


    consoleHandler = logging.StreamHandler()
    consoleHandler.setFormatter(logFormatter)
    logger.addHandler(consoleHandler)

    logger.setLevel(logging.INFO)
    logging.info(pformat(args.__dict__))


    ### 2. set the clean train data and clean test data


    ### 3. generate dataset for backdoor defense and evaluation
    benign_train_ds = train_ds
    benign_test_ds = test_ds
    adv_test_dataset = bd_test_ds

    train_data = DataLoader(
            dataset = benign_train_ds,
            batch_size=args.batch_size,
            shuffle=True,
            drop_last=False,
        )
    
    test_dataset_dict={
                "test_data" :benign_test_ds,
                "adv_test_data" :adv_test_dataset,
        }

    test_dataloader_dict = {
            name : DataLoader(
                    dataset = test_dataset,
                    batch_size=args.batch_size,
                    shuffle=False,
                    drop_last=False,
                )
            for name, test_dataset in test_dataset_dict.items()
        }   


    for dl_name, test_dataloader in test_dataloader_dict.items():
        metrics = test(net, test_dataloader, device)
        metric_info = {
            f'{dl_name} acc': metrics['test_correct'] / metrics['test_total'],
            f'{dl_name} loss': metrics['test_loss'],
        }
        if 'test_data' == dl_name:
            cur_clean_acc = metric_info['test_data acc']
        if 'adv_test_data' == dl_name:
            cur_adv_acc = metric_info['adv_test_data acc']
    logging.info('*****************************')
    logging.info(f"Load from {args.folder_path + '/attack_result.pt'}")
    logging.info(f'Fine-tunning mode: {args.ft_mode}')
    logging.info('Original performance')
    logging.info(f"Test Set: Clean ACC: {cur_clean_acc} | ASR: {cur_adv_acc}")
    logging.info('*****************************')


    original_linear_norm = torch.norm(eval(f'net.{args.linear_name}.weight'))
    weight_mat_ori = eval(f'net.{args.linear_name}.weight.data.clone().detach()')

    param_list = []
    for name, param in net.named_parameters():
        if args.linear_name in name:
            if init:
                if 'weight' in name:
                    logging.info(f'Initialize linear classifier weight {name}.')
                    std = 1 / math.sqrt(param.size(-1)) 
                    param.data.uniform_(-std, std)
                    
                else:
                    logging.info(f'Initialize linear classifier weight {name}.')
                    param.data.uniform_(-std, std)
        if args.ft_mode == 'lp':
            if args.linear_name in name:
                param.requires_grad = True
                param_list.append(param)
            else:
                param.requires_grad = False
        elif args.ft_mode == 'ft' or args.ft_mode == 'fst' or args.ft_mode == 'ft-init':
            param.requires_grad = True
            param_list.append(param)
        elif args.ft_mode == 'fe-tuning':
            if args.linear_name not in name:
                param.requires_grad = True
                param_list.append(param)
            else:
                param.requires_grad = False
        
        

    optimizer = optim.SGD(param_list, lr=args.lr,momentum = 0.9)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)
    criterion = nn.CrossEntropyLoss()
    
    for epoch in range(args.epochs):
        
        batch_loss_list = []
        train_correct = 0
        train_tot = 0
        
    
        logging.info(f'Epoch: {epoch}')
        net.train()

        for batch_idx, (x, labels, *additional_info) in enumerate(train_data):

            
            x, labels = x.to(device), labels.to(device)
            log_probs= net(x)
            if args.lb_smooth is not None:
                loss = lbs_criterion(log_probs, labels)
            else:
                if args.ft_mode == 'fst':
                    loss = torch.sum(eval(f'net.{args.linear_name}.weight') * weight_mat_ori)*args.alpha + criterion(log_probs, labels.long())
                else:
                    loss = criterion(log_probs, labels.long())
            loss.backward()
            
            
            optimizer.step()
            optimizer.zero_grad()

            exec_str = f'net.{args.linear_name}.weight.data = net.{args.linear_name}.weight.data * original_linear_norm  / torch.norm(net.{args.linear_name}.weight.data)'
            exec(exec_str)

            _, predicted = torch.max(log_probs, -1)
            train_correct += predicted.eq(labels).sum()
            train_tot += labels.size(0)
            batch_loss = loss.item() * labels.size(0)
            batch_loss_list.append(batch_loss)

    
        scheduler.step()
        one_epoch_loss = sum(batch_loss_list)


        logging.info(f'Training ACC: {train_correct/train_tot} | Training loss: {one_epoch_loss}')
        logging.info(f'Learning rate: {optimizer.param_groups[0]["lr"]}')
        logging.info('-------------------------------------')
        
        if epoch <= args.epochs-1:
            for dl_name, test_dataloader in test_dataloader_dict.items():
                metrics = test(net, test_dataloader, device)
                metric_info = {
                    f'{dl_name} acc': metrics['test_correct'] / metrics['test_total'],
                    f'{dl_name} loss': metrics['test_loss'],
                }
                if 'test_data' == dl_name:
                    cur_clean_acc = metric_info['test_data acc']
                if 'adv_test_data' == dl_name:
                    cur_adv_acc = metric_info['adv_test_data acc']
            logging.info('Defense performance')
            logging.info(f"Clean ACC: {cur_clean_acc} | ASR: {cur_adv_acc}") 
            logging.info('-------------------------------------')
    
    if args.save:
        model_save_path = f'{target_folder}/ft/'
        torch.save(net.state_dict(), f'{model_save_path}/checkpoint.pt')
        
    
if __name__ == '__main__':
    main()
    