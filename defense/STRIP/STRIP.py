import torch
import os
import torchvision
import numpy as np
import cv2
import torch.nn.functional as F
from torchvision import transforms
import argparse
import sys
sys.path.append('../../')
from tools.utils import manual_seed, get_model, rm_if_exist
from tools.dataset import get_dataset_class_and_scale, get_dataloader, get_de_normalization, get_dataset_normalization
from omegaconf import OmegaConf
# from tools.inject_backdoor import patch_trigger
from tools.img import tensor2ndarray, ndarray2tensor
import matplotlib.pylab as plt
from tqdm import tqdm
from matplotlib.font_manager import FontProperties



def get_argument():
    parser = argparse.ArgumentParser()

    # Directory option
    parser.add_argument("--data_root", type=str, default="/home/ubuntu/temps")
    parser.add_argument("--checkpoints", type=str, default="../../checkpoints")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--results", type=str, default="./results")
    parser.add_argument("--dataset", type=str, default="cifar10")
    parser.add_argument("--attack_mode", type=str, default="all2one")
    parser.add_argument("--temps", type=str, default="./temps")


    parser.add_argument('--path', type=str)

    # ---------------------------- For Neural Cleanse --------------------------
    # Model hyperparameters
    # Model hyperparameters
    parser.add_argument("--n_sample", type=int, default=100)
    parser.add_argument("--n_test", type=int, default=100)
    parser.add_argument("--detection_boundary", type=float, default=0.2)  # According to the original paper
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--test_rounds", type=int, default=10)

    parser.add_argument("--s", type=float, default=0.5)
    parser.add_argument("--k", type=int, default=4)  # low-res grid size
    parser.add_argument(
        "--grid-rescale", type=float, default=1
    )  # scale grid values to avoid going out of [-1, 1]. For example, grid-rescale = 0.98

    return parser

class STRIP:
    def _superimpose(self, background, overlay):
        background = tensor2ndarray(background)
        overlay = tensor2ndarray(overlay)
        output = cv2.addWeighted(background, 1, overlay, 1, 0)
        if len(output.shape) == 2:
            output = np.expand_dims(output, 2)
        return output

    def _get_entropy(self, background, dataset, classifier):
        entropy_sum = [0] * self.n_sample
        x1_add = [0] * self.n_sample
        index_overlay = np.random.randint(0, len(dataset), size=self.n_sample)
        for index in range(self.n_sample):
            ele = dataset[index_overlay[index]][0]
            add_image = self._superimpose(background, ele)
            add_image = ndarray2tensor(add_image)
            x1_add[index] = add_image

        py1_add = classifier(torch.stack(x1_add).to(self.device))
        py1_add = torch.sigmoid(py1_add).detach().cpu().numpy()
        entropy_sum = -np.nansum(py1_add * np.log2(py1_add))
        return entropy_sum / self.n_sample

    def __init__(self, opt):
        super().__init__()
        self.n_sample = opt.n_sample
        self.device = opt.device


    def __call__(self, background, dataset, classifier):
        return self._get_entropy(background, dataset, classifier)

def strip(opt, config, test_dataloader, patch_trigger, testset, netC):
    # STRIP detector
    strip_detector = STRIP(opt)

    # Entropy list
    list_entropy_trojan = []
    list_entropy_benign = []

    de_norm = get_de_normalization(config.dataset_name)
    do_norm = get_dataset_normalization(config.dataset_name)

    mode = 'clean' if config.attack.name == "benign" else 'attack'

    if mode == "attack":
        # Testing with perturbed data
        # print("Testing with bd data !!!!")
        inputs, targets = next(iter(test_dataloader))
        inputs = de_norm(inputs)
        bd_inputs = []
        for i in range(inputs.shape[0]):
            p = patch_trigger(inputs[i], config)
            # p = inputs[i]
            bd_inputs.append(p)
            if len(bd_inputs) >= opt.n_test:
                break
        bd_inputs = torch.stack(bd_inputs, dim=0)
        bd_inputs = bd_inputs.clip_(0, 1)
        bd_inputs = do_norm(bd_inputs)
        for index in range(opt.n_test):
            background = bd_inputs[index]
            entropy = strip_detector(background, testset, netC)
            list_entropy_trojan.append(entropy)

        # Testing with clean data
        for index in range(opt.n_test):
            background, _ = testset[index]
            entropy = strip_detector(background, testset, netC)
            list_entropy_benign.append(entropy)
    else:
        # Testing with clean data
        print("Testing with clean data !!!!")
        for index in range(opt.n_test):
            background, _ = testset[index]
            entropy = strip_detector(background, testset, netC)
            list_entropy_benign.append(entropy)
            # progress_bar(index, opt.n_test)

    return list_entropy_trojan, list_entropy_benign


def main():
    opt = get_argument().parse_args()
    num_class, scale = get_dataset_class_and_scale(opt.dataset)
    opt.input_height = opt.input_width  = scale
    opt.input_channel = 3
    opt.num_classes = num_class

    target_folder = opt.path
    sys.path.append('./run')
    sys.path.append(target_folder)
    from inject_backdoor import patch_trigger
    path = f'{target_folder}/config.yaml'
    config = OmegaConf.load(path)
    manual_seed(config.seed)
    device = f'cuda:{config.device}'
    num_class, scale = get_dataset_class_and_scale(config.dataset_name)
    net = get_model(config.model, num_class, device=device)
    ld = torch.load(f'{target_folder}/results.pth', map_location=device)
    net.load_state_dict(ld['model'])
    net.to(device)
    if config.model == "repvgg":
        net.deploy =True
    _, test_dataloader = get_dataloader(config.dataset_name, 1024, config.pin_memory, config.num_workers)
    testset = test_dataloader.dataset
    netC = net

    lists_entropy_trojan = []
    lists_entropy_benign = []
    for _ in tqdm(range(opt.test_rounds)):
        list_entropy_trojan, list_entropy_benign = strip(opt, config, test_dataloader, patch_trigger, testset, netC)
        lists_entropy_trojan += list_entropy_trojan
        lists_entropy_benign += list_entropy_benign

    # Save result to file
    result_path = os.path.join("{}_{}_output.txt".format(opt.dataset, opt.attack_mode))
    result_path = f'{opt.path}/STRIP/{result_path}'

    rm_if_exist(f'{opt.path}/STRIP/')
    os.makedirs(f'{opt.path}/STRIP/', exist_ok=True)

    min_entropy = min(lists_entropy_trojan + lists_entropy_benign)

    with open(result_path, "w+") as f:
        for index in range(len(lists_entropy_trojan)):
            if index < len(lists_entropy_trojan) - 1:
                f.write("{} ".format(lists_entropy_trojan[index]))
            else:
                f.write("{}".format(lists_entropy_trojan[index]))

        f.write("\n")

        for index in range(len(lists_entropy_benign)):
            if index < len(lists_entropy_benign) - 1:
                f.write("{} ".format(lists_entropy_benign[index]))
            else:
                f.write("{}".format(lists_entropy_benign[index]))
        f.write(f"\n{min_entropy}")
    # Determining
    print("Min entropy trojan: {}, Detection boundary: {}".format(min_entropy, opt.detection_boundary))
    if min_entropy < opt.detection_boundary:
        print("A backdoored model\n")
    else:
        print("Not a backdoor model\n")

    print(f'min benign: {min(lists_entropy_benign)}, len: {len(lists_entropy_benign)}')
    print(f'min trojaned: {min(lists_entropy_trojan)}, len: {len(lists_entropy_trojan)}')

    # 设置字体为Calibri
    font = FontProperties(fname='/home/chengyiqiu/code/INBA/resource/Fonts/Calibri.ttf', size=20)

    # 假设 lists_entropy_benign 和 lists_entropy_trojan 已经被定义并包含数据

    N = len(lists_entropy_benign)
    bins_sturges = int(np.ceil(np.log2(N) + 1))  # Sturges' Rule
    bins_rice = int(np.ceil(2 * (N ** (1 / 3))))  # Rice Rule

    bins = bins_rice
    alpha = 0.9  # 设置透明度
    rwidth = 0.9  # 设置柱子的相对宽度，小于1会在柱子之间创建间隔

    # 绘制直方图
    plt.hist(lists_entropy_benign, bins=bins, weights=np.ones(len(lists_entropy_benign)) / len(lists_entropy_benign), alpha=alpha, label='without trojan', rwidth=rwidth)
    plt.hist(lists_entropy_trojan, bins=bins, weights=np.ones(len(lists_entropy_trojan)) / len(lists_entropy_trojan), alpha=alpha, label='with trojan', rwidth=rwidth)

    # 设置图例、轴标签和标题
    # plt.legend(loc='upper right', prop=font)
    plt.xlabel('Entropy', fontproperties=font)
    plt.ylabel('Probability', fontproperties=font)
    plt.title('Normalized Entropy', fontproperties=font)
    plt.tick_params(labelsize=20)

    plt.tight_layout()

    # 保存图像
    fig1 = plt.gcf()
    fig1.savefig(f'{opt.path}/STRIP/Entropy.png', dpi=500)

    # 显示图形
    plt.show()

if __name__ == "__main__":
    main()