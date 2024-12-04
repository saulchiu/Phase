import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
from omegaconf import OmegaConf
import argparse
import os

sys.path.append('../')
from tools.dataset import get_dataloader, get_de_normalization, get_dataset_normalization, get_dataset_class_and_scale
from tools.utils import manual_seed, rm_if_exist, get_model
from tools.img import tensor2ndarray

# 解析参数
parser = argparse.ArgumentParser('')
parser.add_argument('--path', type=str, default='/home/chengyiqiu/code/INBA/results/cifar10/badnet/presnet18/20241129035748')
parser.add_argument('--label', type=int, default=1)
args = parser.parse_args()

# 加载配置
target_folder = args.path
path = f'{target_folder}/config.yaml'
config = OmegaConf.load(path)
manual_seed(config.seed)
device = f'cuda:{config.device}'
num_classes, scale = get_dataset_class_and_scale(config.dataset_name)

# 根据模型配置选择网络架构
net = get_model(config.model, num_classes, device=device)
ld = torch.load(f'{target_folder}/results.pth', map_location=device)
net.load_state_dict(ld['model'])
net.to(device)
if config.model == "repvgg":
    net.deploy =True
train_dl, test_dl = get_dataloader(config.dataset_name, config.batch, config.pin_memory, config.num_workers)
x_c = None

# 获取目标类的图像
for batch, label in train_dl:
    batch = batch.to(device)
    for i in range(batch.shape[0]):
        if label[i].item() == args.label:
            x_c = batch[i]
            break
    if x_c is None:
        continue


# 获取模型预测
target = 1
y_c_logits, outs = net(x_c.unsqueeze(0))
out4 = outs[target]
print(out4.shape)
out4 = out4.detach().cpu()
out4.requires_grad_(False)
out4.resize_(1, 3, 32, 32)
out4.squeeze_()
print(out4.shape)
out4 = tensor2ndarray(out4)

de_norm = get_de_normalization(config.dataset_name)
do_norm = get_dataset_normalization(config.dataset_name)
sys.path.append(target_folder)
from inject_backdoor import patch_trigger
x_p = patch_trigger(de_norm(x_c).squeeze(), config)
x_p.clip_(0, 1)
x_p = do_norm(x_p)
x_p = x_p.to(device)
y_c_logits, outs = net(x_p.unsqueeze(0))
out_p = outs[target]
print(out_p.shape)
out_p = out_p.detach().cpu()
out_p.requires_grad_(False)
out_p.resize_(1, 3, 32, 32)
out_p.squeeze_()
print(out_p.shape)
out_p = tensor2ndarray(out_p)


_, axs = plt.subplots(1, 1)
axs.imshow(out_p)
rm_if_exist(f'{target_folder}/residual_analyze/')
os.makedirs(f'{target_folder}/residual_analyze', exist_ok=True)
plt.savefig(f'{target_folder}/residual_analyze/residuals_heatmap.png')
plt.show()
