import sys
sys.path.append('/home/chengyiqiu/code/INBA/')
import PIL.Image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
from torchvision.models import resnet50
from tools.dataset import get_dataloader, get_dataset_normalization
from tools.img import tensor2ndarray, ndarray2tensor
import matplotlib.pyplot as plt
from tools.dataset import get_de_normalization, get_dataset_class_and_scale, get_dataset_normalization, get_train_and_test_dataset
from omegaconf import OmegaConf, DictConfig
from tools.utils import manual_seed, rm_if_exist
from torchvision.transforms.transforms import ToTensor, Resize, Compose
import torch
import random
import PIL
from tools.dataset import get_benign_transform
import numpy as np
import os
import argparse
from tools.utils import get_model

parser = argparse.ArgumentParser('')
parser.add_argument(
    '--path',
    type=str,
    default='/home/chengyiqiu/code/INBA/results/imagenette/inba/20241126132759'
)
parser.add_argument(
    '--label',
    type=int,
    default=1
)
args = parser.parse_args()

from pytorch_grad_cam import GradCAM, HiResCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, XGradCAM, EigenCAM, FullGrad
cam_class = GradCAMPlusPlus

target_folder = args.path
path = f'{target_folder}/config.yaml'
config = OmegaConf.load(path)
manual_seed(config.seed)
device = f'cuda:{config.device}' 
num_classes, scale = get_dataset_class_and_scale(config.dataset_name)
target_class = args.label

net = get_model(config.model, num_classes, device=device)
ld = torch.load(f'{target_folder}/results.pth', map_location=device)
net.load_state_dict(ld['model'])
net.to(device)
net.eval()
target_layers = [net.layer4[-1]]

x_c = None
ctr = 6
# train_data, test_data = get_train_and_test_dataset(config.dataset_name)
# for (x, y) in train_data:
#     if y == target_class:
#         if ctr <= 0:
#             x_c = x
#             break
#         else:
#             ctr -= 1
# x_c = x_c.to(device)

x_c = PIL.Image.open('/home/chengyiqiu/code/INBA/data/imagenette2/train/n01440764/ILSVRC2012_val_00006697.JPEG')
x_c = Compose([ToTensor(), Resize((scale, scale)), get_dataset_normalization(config.dataset_name)])(x_c)
x_c = x_c.to(device)


cam = cam_class(model=net, target_layers=target_layers)
y_c = net(x_c.unsqueeze(0))
_, y_c = torch.max(y_c, 1)
benign_heat: np.ndarray = cam(x_c.unsqueeze(0), targets=[ClassifierOutputTarget(y_c.item())])


net = get_model(config.model, num_classes, device=device)
ld = torch.load(f'{target_folder}/results.pth', map_location=device)
net.load_state_dict(ld['model'])
net.to(device)
net.eval()
target_layers = [net.layer4[-1]]

de_norm = get_de_normalization(config.dataset_name)
do_norm = get_dataset_normalization(config.dataset_name)

sys.path.append(target_folder)
from inject_backdoor import patch_trigger
x_p = patch_trigger(de_norm(x_c).squeeze(), config)
x_p.clip_(0, 1)
x_p = do_norm(x_p)
x_p = x_p.to(device)
cam = cam_class(model=net, target_layers=target_layers)
y_p = net(x_p.unsqueeze(0))
_, y_p = torch.max(y_p, 1)
bd_heat: np.ndarray = cam(x_p.unsqueeze(0), targets=[ClassifierOutputTarget(y_p.item())])



visualization_0 = show_cam_on_image(tensor2ndarray(get_de_normalization(config.dataset_name)(x_c).squeeze()) / 255.,
                                     benign_heat[0, :], use_rgb=True)
x_p = de_norm(x_p).squeeze()
visualization_1 = show_cam_on_image(tensor2ndarray(x_p) / 255., bd_heat[0, :], use_rgb=True)

_, axs = plt.subplots(2, 2, figsize=(15, 10))
axs[0, 0].imshow(tensor2ndarray(get_de_normalization(config.dataset_name)(x_c).squeeze()))
axs[0, 0].set_title(f'x_c, label: {y_c.item()}')
axs[0, 0].axis('off')


axs[0, 1].imshow(tensor2ndarray(x_p))
axs[0, 1].set_title(f'x_p, label: {y_p.item()}')
axs[0, 1].axis('off')

axs[1, 0].imshow(visualization_0)
axs[1, 0].set_title('benign_heat')
axs[1, 0].axis('off')

axs[1, 1].imshow(visualization_1)
axs[1, 1].set_title('poison_heat')
axs[1, 1].axis('off')

rm_if_exist(f'{target_folder}/GradCam/')
os.makedirs(f'{target_folder}/GradCam', exist_ok=True)
plt.savefig(f'{target_folder}/GradCam/hotmap.png')
plt.show()

PIL.Image.Image.save(PIL.Image.fromarray(visualization_1), f'{target_folder}/GradCam/singal.png')
PIL.Image.Image.save(PIL.Image.fromarray(tensor2ndarray(get_de_normalization(config.dataset_name)(x_c).squeeze())), f'{target_folder}/GradCam/raw.png')


