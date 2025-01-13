import sys
sys.path.append('../')

import os
import time
import argparse
import logging
import numpy as np
import torch
import torch.nn as nn
import pandas as pd
from collections import OrderedDict
import classifier_models
from tools.dataset import get_train_and_test_dataset, PartialDataset, PoisonDataset, get_dataset_class_and_scale
from tools.utils import get_model
from torch.utils.data.dataloader import DataLoader

if torch.cuda.is_available():
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

seed = 98
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.manual_seed(seed)
np.random.seed(seed)


def train_step_unlearning(args, model, criterion, optimizer, data_loader):
    model.train()
    total_correct = 0
    total_loss = 0.0
    for i, (images, labels) in enumerate(data_loader):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        output = model(images)
        loss = criterion(output, labels)

        pred = output.data.max(1)[1]
        total_correct += pred.eq(labels.view_as(pred)).sum()
        total_loss += loss.item()

        nn.utils.clip_grad_norm_(model.parameters(), max_norm=20, norm_type=2)
        (-loss).backward()
        optimizer.step()

    loss = total_loss / len(data_loader)
    acc = float(total_correct) / len(data_loader.dataset)
    return loss, acc

def train_step_recovering(args, unlearned_model, criterion, mask_opt, data_loader):
    unlearned_model.train()
    total_correct = 0
    total_loss = 0.0
    nb_samples = 0
    for i, (images, labels) in enumerate(data_loader):
        images, labels = images.to(device), labels.to(device)
        nb_samples += images.size(0)

        mask_opt.zero_grad()
        output = unlearned_model(images)
        loss = criterion(output, labels)
        loss = args.alpha * loss

        pred = output.data.max(1)[1]
        total_correct += pred.eq(labels.view_as(pred)).sum()
        total_loss += loss.item()
        loss.backward()
        mask_opt.step()
        clip_mask(unlearned_model)

    loss = total_loss / len(data_loader)
    acc = float(total_correct) / nb_samples
    return loss, acc


def load_state_dict(net, orig_state_dict):
    if 'state_dict' in orig_state_dict.keys():
        orig_state_dict = orig_state_dict['state_dict']

    new_state_dict = OrderedDict()
    for k, v in net.state_dict().items():
        if k in orig_state_dict.keys():
            new_state_dict[k] = orig_state_dict[k]
        else:
            new_state_dict[k] = v
    net.load_state_dict(new_state_dict)


def clip_mask(unlearned_model, lower=0.0, upper=1.0):
    params = [param for name, param in unlearned_model.named_parameters() if 'neuron_mask' in name]
    with torch.no_grad():
        for param in params:
            param.clamp_(lower, upper)


def save_mask_scores(state_dict, file_name):
    mask_values = []
    count = 0
    for name, param in state_dict.items():
        if 'neuron_mask' in name:
            for idx in range(param.size(0)):
                neuron_name = '.'.join(name.split('.')[:-1])
                mask_values.append('{} \t {} \t {} \t {:.4f} \n'.format(count, neuron_name, idx, param[idx].item()))
                count += 1
    with open(file_name, "w") as f:
        f.write('No \t Layer Name \t Neuron Idx \t Mask Score \n')
        f.writelines(mask_values)

def read_data(file_name):
    tempt = pd.read_csv(file_name, sep='\s+', skiprows=1, header=None)
    layer = tempt.iloc[:, 1]
    idx = tempt.iloc[:, 2]
    value = tempt.iloc[:, 3]
    mask_values = list(zip(layer, idx, value))
    return mask_values

def pruning(net, neuron):
    state_dict = net.state_dict()
    weight_name = '{}.{}'.format(neuron[0], 'weight')
    state_dict[weight_name][int(neuron[1])] = 0.0
    net.load_state_dict(state_dict)

def test(model, criterion, data_loader):
    model.eval()
    total_correct = 0
    total_loss = 0.0
    with torch.no_grad():
        for i, (images, labels) in enumerate(data_loader):
            images, labels = images.to(device), labels.to(device)
            output = model(images)
            total_loss += criterion(output, labels).item()
            pred = output.data.max(1)[1]
            total_correct += pred.eq(labels.data.view_as(pred)).sum()
    loss = total_loss / len(data_loader)
    acc = float(total_correct) / len(data_loader.dataset)
    return loss, acc

def evaluate_by_number(model, logger, mask_values, pruning_max, pruning_step, criterion, clean_loader, poison_loader):
    results = []
    nb_max = int(np.ceil(pruning_max))
    nb_step = int(np.ceil(pruning_step))
    for start in range(0, nb_max + 1, nb_step):
        i = start
        for i in range(start, start + nb_step):
            pruning(model, mask_values[i])
        layer_name, neuron_idx, value = mask_values[i][0], mask_values[i][1], mask_values[i][2]
        cl_loss, cl_acc = test(model=model, criterion=criterion, data_loader=clean_loader)
        po_loss, po_acc = test(model=model, criterion=criterion, data_loader=poison_loader)
        logger.info('{} \t {} \t {} \t {} \t {:.4f} \t {:.4f} \t {:.4f} \t {:.4f}'.format(
            i+1, layer_name, neuron_idx, value, po_loss, po_acc, cl_loss, cl_acc))
        results.append('{} \t {} \t {} \t {} \t {:.4f} \t {:.4f} \t {:.4f} \t {:.4f}'.format(
            i+1, layer_name, neuron_idx, value, po_loss, po_acc, cl_loss, cl_acc))
    return results


def evaluate_by_threshold(model, logger, mask_values, pruning_max, pruning_step, criterion, clean_loader, poison_loader):
    results = []
    thresholds = np.arange(0, pruning_max + pruning_step, pruning_step)
    start = 0
    for threshold in thresholds:
        idx = start
        for idx in range(start, len(mask_values)):
            if float(mask_values[idx][2]) <= threshold:
                pruning(model, mask_values[idx])
                start += 1
            else:
                break
        layer_name, neuron_idx, value = mask_values[idx][0], mask_values[idx][1], mask_values[idx][2]
        cl_loss, cl_acc = test(model=model, criterion=criterion, data_loader=clean_loader)
        po_loss, po_acc = test(model=model, criterion=criterion, data_loader=poison_loader)
        logger.info('{:.2f} \t {} \t {} \t {} \t {:.4f} \t {:.4f} \t {:.4f} \t {:.4f}'.format(
            start, layer_name, neuron_idx, threshold, po_loss, po_acc, cl_loss, cl_acc))
        results.append('{:.2f} \t {} \t {} \t {} \t {:.4f} \t {:.4f} \t {:.4f} \t {:.4f}\n'.format(
            start, layer_name, neuron_idx, threshold, po_loss, po_acc, cl_loss, cl_acc))
    return results, cl_acc, po_acc

def save_checkpoint(state, file_path):
    # filepath = os.path.join(args.output_dir, args.arch + '-unlearning_epochs{}.tar'.format(epoch))
    torch.save(state, file_path)


from omegaconf import OmegaConf
from tools.utils import manual_seed
from classifier_models.defense.mask_batchnorm import MaskBatchNorm2d

def main(args):
    target_folder = args.path
    device = f'cuda:{args.cuda}'
    path = f'{target_folder}/config.yaml'
    config = OmegaConf.load(path)
    manual_seed(66)
    config.attack.mode = 'defense'
    args.log_root = f'{target_folder}/rnp/log'
    args.backdoor_model_path = args.output_weight = f'{target_folder}/rnp/'
    os.makedirs(args.log_root, exist_ok=True)

    logger = logging.getLogger(__name__)
    logging.basicConfig(
        format='[%(asctime)s] - %(message)s',
        datefmt='%Y/%m/%d %H:%M:%S',
        level=logging.DEBUG,
        handlers=[
            logging.FileHandler(os.path.join(args.log_root, 'output.log')),
            logging.StreamHandler()
        ])
    logger.info(args)

    logger.info('----------- Data Initialization --------------')
    # defense_data_loader = get_train_loader(args)
    # clean_test_loader, bad_test_loader = get_test_loader(args)
    train_ds, test_ds = get_train_and_test_dataset(config.dataset_name)
    partial_ds = PartialDataset(train_ds, args.ratio)
    poison_config = config.copy()
    poison_config.ratio = 1
    bad_test_ds = PoisonDataset(test_ds, poison_config)

    defense_data_loader = DataLoader(partial_ds, config.batch, True, num_workers=4, drop_last=True)
    clean_test_loader = DataLoader(test_ds, config.batch, False, num_workers=4, drop_last=False)
    bad_test_loader = DataLoader(bad_test_ds, config.batch, False, num_workers=4, drop_last=False)

    num_class, scale = get_dataset_class_and_scale(config.dataset_name)
    logger.info('----------- Backdoor Model Initialization --------------')
    from classifier_models.defense.RNP_model import resnet18
    net = resnet18(num_classes=num_class, norm_layer=None).to(device=device)
    ld = torch.load(f'{target_folder}/results.pth', map_location=device)
    net.load_state_dict(ld['model'])
    net.to(device=device)

    criterion = torch.nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.SGD(net.parameters(), lr=args.unlearning_lr, momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.schedule, gamma=0.1)

    logger.info('----------- Model Unlearning --------------')
    logger.info('Epoch \t lr \t Time \t TrainLoss \t TrainACC \t PoisonLoss \t PoisonACC \t CleanLoss \t CleanACC')
    for epoch in range(0, args.unlearning_epochs + 1):
        start = time.time()
        lr = optimizer.param_groups[0]['lr']
        train_loss, train_acc = train_step_unlearning(args=args, model=net, criterion=criterion, optimizer=optimizer,
                                      data_loader=defense_data_loader)
        cl_test_loss, cl_test_acc = test(model=net, criterion=criterion, data_loader=clean_test_loader)
        po_test_loss, po_test_acc = test(model=net, criterion=criterion, data_loader=bad_test_loader)
        scheduler.step()
        end = time.time()
        logger.info(
            '%d \t %.3f \t %.1f \t %.4f \t %.4f \t %.4f \t %.4f \t %.4f \t %.4f',
            epoch, lr, end - start, train_loss, train_acc, po_test_loss, po_test_acc,
            cl_test_loss, cl_test_acc)

        if train_acc <= args.clean_threshold:
            # save the last checkpoint
            args.output_weight = f'{config.path}/rnp/output_weight'
            os.makedirs(args.output_weight, exist_ok=True)
            file_path = os.path.join(args.output_weight, f'unlearned_model_last.tar')
            save_checkpoint({
                'epoch': epoch,
                'state_dict': net.state_dict(),
                'clean_acc': cl_test_acc,
                'bad_acc': po_test_acc,
                'optimizer': optimizer.state_dict(),
            }, file_path)
            break


    logger.info('----------- Model Recovering --------------')
    # Step 2: load unleanred model checkpoints
    if args.unlearned_model_path is not None:
        unlearned_model_path = args.unlearned_model_path
    else:
        unlearned_model_path = os.path.join(args.output_weight, 'unlearned_model_last.tar')

    checkpoint = torch.load(unlearned_model_path, map_location=device)
    print('Unlearned Model:', checkpoint['epoch'], checkpoint['clean_acc'], checkpoint['bad_acc'])
    args.arch = config.model

    from classifier_models.defense.RNP_model import resnet18
    unlearned_model = resnet18(num_classes=num_class, norm_layer=MaskBatchNorm2d).to(device)

    # unlearned_model = getattr(models, args.arch)(num_classes=10, norm_layer=models.MaskBatchNorm2d)
    load_state_dict(unlearned_model, orig_state_dict=checkpoint['state_dict'])
    unlearned_model = unlearned_model.to(device)
    criterion = torch.nn.CrossEntropyLoss().to(device)

    parameters = list(unlearned_model.named_parameters())
    mask_params = [v for n, v in parameters if "neuron_mask" in n]
    mask_optimizer = torch.optim.SGD(mask_params, lr=args.recovering_lr, momentum=0.9)

    # Recovering
    logger.info('Epoch \t lr \t Time \t TrainLoss \t TrainACC \t PoisonLoss \t PoisonACC \t CleanLoss \t CleanACC')
    for epoch in range(1, args.recovering_epochs + 1):
        start = time.time()
        lr = mask_optimizer.param_groups[0]['lr']
        train_loss, train_acc = train_step_recovering(args=args, unlearned_model=unlearned_model, criterion=criterion, data_loader=defense_data_loader,
                                           mask_opt=mask_optimizer)
        cl_test_loss, cl_test_acc = test(model=unlearned_model, criterion=criterion, data_loader=clean_test_loader)
        po_test_loss, po_test_acc = test(model=unlearned_model, criterion=criterion, data_loader=bad_test_loader)
        end = time.time()
        logger.info('{} \t {:.3f} \t {:.1f} \t {:.4f} \t {:.4f} \t {:.4f} \t {:.4f} \t {:.4f} \t {:.4f}'.format(
            epoch, lr, end - start, train_loss, train_acc, po_test_loss, po_test_acc,
            cl_test_loss, cl_test_acc))
    save_mask_scores(unlearned_model.state_dict(), os.path.join(args.log_root, 'mask_values.txt'))

    del unlearned_model, net
    logger.info('----------- Backdoored Model Pruning --------------')
    # load model checkpoints and trigger info
    # state_dict = torch.load(args.backdoor_model_path, map_location=device)
    # net = getattr(models, args.arch)(num_classes=10, norm_layer=None)
    # load_state_dict(net, orig_state_dict=state_dict)
    # net = net.to(device)
    net = get_model(config.model, num_class, device=device)
    ld = torch.load(f'{target_folder}/results.pth', map_location=device)
    net.load_state_dict(ld['model'])
    net.to(device)
    criterion = torch.nn.CrossEntropyLoss().to(device)

    # Step 3: pruning
    if args.mask_file is not None:
        mask_file = args.mask_file
    else:
        mask_file = os.path.join(args.log_root, 'mask_values.txt')

    mask_values = read_data(mask_file)
    mask_values = sorted(mask_values, key=lambda x: float(x[2]))
    logger.info('No. \t Layer Name \t Neuron Idx \t Mask \t PoisonLoss \t PoisonACC \t CleanLoss \t CleanACC')
    cl_loss, cl_acc = test(model=net, criterion=criterion, data_loader=clean_test_loader)
    po_loss, po_acc = test(model=net, criterion=criterion, data_loader=bad_test_loader)
    logger.info('0 \t None     \t None     \t {:.4f} \t {:.4f} \t {:.4f} \t {:.4f}'.format(po_loss, po_acc, cl_loss, cl_acc))
    if args.pruning_by == 'threshold':
        results, clean_acc, poisoned_acc = evaluate_by_threshold(
            net, logger, mask_values, pruning_max=args.pruning_max, pruning_step=args.pruning_step,
            criterion=criterion, clean_loader=clean_test_loader, poison_loader=bad_test_loader
        )
    else:
        results = evaluate_by_number(
            net, logger, mask_values, pruning_max=args.pruning_max, pruning_step=args.pruning_step,
            criterion=criterion, clean_loader=clean_test_loader, poison_loader=bad_test_loader
        )
    file_name = os.path.join(args.log_root, 'pruning_by_{}.txt'.format(args.pruning_by))
    with open(file_name, "w") as f:
        f.write('No \t Layer Name \t Neuron Idx \t Mask \t PoisonLoss \t PoisonACC \t CleanLoss \t CleanACC\n')
        f.writelines(results)
    return clean_acc, poisoned_acc


if __name__ == '__main__':
    # Prepare arguments
    parser = argparse.ArgumentParser()

    # various path
    parser.add_argument('--cuda', type=int, default=0, help='cuda available')
    parser.add_argument('--save-every', type=int, default=5, help='save checkpoints every few epochs')
    parser.add_argument('--log_root', type=str, default='logs/', help='logs are saved here')
    parser.add_argument('--output_weight', type=str, default='weights/')
    parser.add_argument('--backdoor_model_path', type=str,
                        default='weights/ResNet18-ResNet-BadNets-target0-portion0.1-epoch80.tar',
                        help='path of backdoored model')
    parser.add_argument('--unlearned_model_path', type=str,
                        default=None, help='path of unlearned backdoored model')
    parser.add_argument('--arch', type=str, default='resnet18',
                        choices=['resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152', 'MobileNetV2',
                                 'vgg19_bn'])
    parser.add_argument('--schedule', type=int, nargs='+', default=[10, 20],
                        help='Decrease learning rate at these epochs.')
    parser.add_argument('--dataset', type=str, default='CIFAR10', help='name of image dataset')
    parser.add_argument('--batch_size', type=int, default=128, help='The size of batch')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
    parser.add_argument('--weight_decay', type=float, default=5e-4, help='weight decay')
    parser.add_argument('--num_class', type=int, default=10, help='number of classes')
    parser.add_argument('--ratio', type=float, default=0.01, help='ratio of defense data')

    # backdoor attacks
    parser.add_argument('--target_label', type=int, default=0, help='class of target label')
    parser.add_argument('--trigger_type', type=str, default='gridTrigger', help='type of backdoor trigger')
    parser.add_argument('--target_type', type=str, default='all2one', help='type of backdoor label')
    parser.add_argument('--trig_w', type=int, default=3, help='width of trigger pattern')
    parser.add_argument('--trig_h', type=int, default=3, help='height of trigger pattern')

    # RNP
    parser.add_argument('--alpha', type=float, default=0.2)
    parser.add_argument('--clean_threshold', type=float, default=0.20, help='threshold of unlearning accuracy')
    parser.add_argument('--unlearning_lr', type=float, default=0.01, help='the learning rate for neuron unlearning')
    parser.add_argument('--recovering_lr', type=float, default=0.2, help='the learning rate for mask optimization')
    parser.add_argument('--unlearning_epochs', type=int, default=20, help='the number of epochs for unlearning')
    parser.add_argument('--recovering_epochs', type=int, default=20, help='the number of epochs for recovering')
    parser.add_argument('--mask_file', type=str, default=None, help='The text file containing the mask values')
    parser.add_argument('--pruning-by', type=str, default='threshold', choices=['number', 'threshold'])
    parser.add_argument('--pruning-max', type=float, default=0.9, help='the maximum number/threshold for pruning')
    parser.add_argument('--pruning-step', type=float, default=0.05, help='the step size for evaluating the pruning')

    parser.add_argument('--path', type=str)

    args = parser.parse_args()
    args_dict = vars(args)
    print(args_dict)
    # os.makedirs(args.log_root, exist_ok=True)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    ratios = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    BA_list = []
    ASR_list = []
    os.makedirs(f'{args.path}/rnp_plot/', exist_ok=True)
    for ratio in ratios:
        args.pruning_max = ratio
        BA, ASR = main(args)
        BA_list.append(BA)
        ASR_list.append(ASR)
        print(BA_list)
        print(ASR_list)
    res = {
        'acc_list': BA_list,
        'asr_list': ASR_list,
    }
    torch.save(res, f'{args.path}/rnp_plot/plot_results.pth')
