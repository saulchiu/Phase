import torch
import os
import torch.nn as nn
import copy
import torch.nn.functional as F
import argparse
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

def create_targets_bd(targets, opt):
    if opt.attack_mode == "all2one":
        bd_targets = torch.ones_like(targets) * opt.target_label
    elif opt.attack_mode == "all2all":
        bd_targets = torch.tensor([(label + 1) % opt.num_classes for label in targets])
    else:
        raise Exception("{} attack mode is not implemented".format(opt.attack_mode))
    return bd_targets.to(opt.device)


def create_bd(netG, netM, inputs, targets, opt):
    bd_targets = create_targets_bd(targets, opt)
    patterns = netG(inputs)
    patterns = netG.normalize_pattern(patterns)

    masks_output = netM.threshold(netM(inputs))
    bd_inputs = inputs + (patterns - inputs) * masks_output
    return bd_inputs, bd_targets


def eval(netC, test_dl, opt, patch_trigger, de_norm, do_norm, config):
    # print(" Eval:")
    acc_clean = 0.0
    acc_bd = 0.0
    total_sample = 0
    total_correct_clean = 0
    total_correct_bd = 0

    for batch_idx, (inputs, targets) in enumerate(test_dl):
        inputs, targets = inputs.to(opt.device), targets.to(opt.device)
        bs = inputs.shape[0]
        total_sample += bs

        # Evaluating clean
        preds_clean = netC(inputs)
        correct_clean = torch.sum(torch.argmax(preds_clean, 1) == targets)
        total_correct_clean += correct_clean
        acc_clean = total_correct_clean * 100.0 / total_sample

        # Evaluating backdoor
        targets_bd = torch.ones_like(targets) * opt.target_label

        inputs = de_norm(inputs)
        bd_inputs = []
        for i in range(inputs.shape[0]):
            p = patch_trigger(inputs[i], config)
            # p = inputs[i]
            bd_inputs.append(p)
            # if len(bd_inputs) >= opt.n_test:
            #     break
        bd_inputs = torch.stack(bd_inputs, dim=0)
        bd_inputs = bd_inputs.clip_(0, 1)
        bd_inputs = do_norm(bd_inputs)

        preds_bd = netC(bd_inputs)
        correct_bd = torch.sum(torch.argmax(preds_bd, 1) == targets_bd)
        total_correct_bd += correct_bd
        acc_bd = total_correct_bd * 100.0 / total_sample
        # progress_bar(batch_idx, len(test_dl), "Acc Clean: {:.3f} | Acc Bd: {:.3f}".format(acc_clean, acc_bd))
    return acc_clean, acc_bd

def get_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_root", type=str, default="/home/ubuntu/temps")
    parser.add_argument("--checkpoints", type=str, default="../../checkpoints")
    parser.add_argument("--temps", type=str, default="./temps")
    parser.add_argument("--device", type=int, default=0)

    parser.add_argument("--dataset", type=str, default="celeba")
    parser.add_argument("--input_height", type=int, default=None)
    parser.add_argument("--input_width", type=int, default=None)
    parser.add_argument("--input_channel", type=int, default=None)
    parser.add_argument("--num_classes", type=int, default=10)

    parser.add_argument("--bs", type=int, default=100)
    parser.add_argument("--num_workers", type=int, default=2)

    parser.add_argument("--attack_mode", type=str, default="all2one", help="all2one or all2all")
    parser.add_argument("--target_label", type=int, default=0)
    parser.add_argument("--outfile", type=str, default="./results.txt")
    parser.add_argument("--k", type=int, default=4)
    parser.add_argument("--s", type=float, default=0.5)
    parser.add_argument("--grid_rescale", type=float, default=1)

    parser.add_argument(
        '--path',
        type=str
    )

    return parser

import sys
sys.path.append('../')
from omegaconf import OmegaConf
from tools.utils import manual_seed, get_model, rm_if_exist
from tools.dataset import get_dataset_class_and_scale, get_dataloader, get_de_normalization, get_dataset_normalization

def main():
    opt = get_arguments().parse_args()
    target_folder = opt.path
    path = f'{target_folder}/config.yaml'
    config = OmegaConf.load(path)

    opt.data_root = opt.checkpoints = f'{target_folder}/fine_pruning/'
    rm_if_exist(opt.data_root)
    os.makedirs(opt.data_root, exist_ok=True)

    manual_seed(config.seed)
    device = f'cuda:{opt.device}'
    num_class, scale = get_dataset_class_and_scale(config.dataset_name)
    net = get_model(config.model, num_class, device=device)
    ld = torch.load(f'{target_folder}/results.pth', map_location=device)
    net.load_state_dict(ld['model'])
    net.to(device)
    _, test_dataloader = get_dataloader(config.dataset_name, 64, config.pin_memory, config.num_workers)
    mode = opt.attack_mode
    netC = net
    netC.eval()
    netC.requires_grad_(False)
    test_dl = test_dataloader

    # Forward hook for getting layer's output
    container = []

    def forward_hook(module, input, output):
        container.append(output)

    hook = netC.layer4.register_forward_hook(forward_hook)

    # Forwarding all the validation set
    print("Forwarding all the validation dataset:")
    for batch_idx, (inputs, _) in enumerate(test_dl):
        inputs = inputs.to(opt.device)
        netC(inputs)
        # progress_bar(batch_idx, len(test_dl))

    # Processing to get the "more important mask"
    container = torch.cat(container, dim=0)
    activation = torch.mean(container, dim=[0, 2, 3])
    seq_sort = torch.argsort(activation)
    pruning_mask = torch.ones(seq_sort.shape[0], dtype=bool)
    hook.remove()

    # Pruning times - no-tuning after pruning a channel!!!
    acc_clean = []
    acc_bd = []
    opt.outfile = f"{opt.data_root}/results.txt"

    do_norm = get_dataset_normalization(config.dataset_name)
    de_norm = get_de_normalization(config.dataset_name)
    sys.path.append(target_folder)
    from inject_backdoor import patch_trigger
    BA_list = []
    ASR_list = []
    print(pruning_mask.shape[0])
    with open(opt.outfile, "w") as outs:
        pbar = tqdm(range(pruning_mask.shape[0]))
        for index in pbar:
            net_pruned = copy.deepcopy(netC)
            num_pruned = index
            if index:
                channel = seq_sort[index - 1]
                pruning_mask[channel] = False
            # print("Pruned {} filters".format(num_pruned))

            net_pruned.layer4[1].conv2 = nn.Conv2d(
                pruning_mask.shape[0], pruning_mask.shape[0] - num_pruned, (3, 3), stride=1, padding=1, bias=False
            )
            net_pruned.linear = nn.Linear(pruning_mask.shape[0] - num_pruned, 10)

            # Re-assigning weight to the pruned net
            for name, module in net_pruned._modules.items():
                if "layer4" in name:
                    module[1].conv2.weight.data = netC.layer4[1].conv2.weight.data[pruning_mask]
                    module[1].ind = pruning_mask
                elif "linear" == name:
                    module.weight.data = netC.linear.weight.data[:, pruning_mask]
                    module.bias.data = netC.linear.bias.data
                else:
                    continue
            net_pruned.to(opt.device)
            clean, bd = eval(net_pruned, test_dl, opt, patch_trigger, de_norm, do_norm, config)
            BA_list.append(clean.item())
            ASR_list.append(bd.item())
            outs.write("%d %0.4f %0.4f\n" % (index, clean, bd))
            pbar.set_postfix(BA=f'{clean:.2f}', ASR=f'{bd:.2f}')
    length = len(BA_list)
    fraction_pruned = np.arange(0, length) / length
    plt.plot(fraction_pruned, BA_list, label='BA')
    plt.plot(fraction_pruned, ASR_list, label='ASR')
    plt.xlabel('Pruning Ratio')
    plt.ylabel('BA / ASR')
    plt.title('Pruning Ratio vs. Accuracy / Attack Success Rate')
    plt.legend()
    plt.ylim(0, 100)
    plt.savefig(f'{opt.data_root}/BA_ASR_Ratio.png')
    plt.show()

    res = {
        'acc_list': BA_list,
        'asr_list': ASR_list,
    }
    torch.save(res, f'{opt.data_root}/plot_results.pth')
    


if __name__ == "__main__":
    main()