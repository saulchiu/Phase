import sys
sys.path.append('../')
import PIL.Image
from pytorch_grad_cam import GradCAM, HiResCAM, GradCAMPlusPlus, XGradCAM, EigenCAM, FullGrad
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
from torchvision.models import resnet50
from tools.dataset import get_dataloader
from tools.img import tensor2ndarray
import matplotlib.pyplot as plt
from tools.dataset import get_de_normalization, get_dataset_class_and_scale, get_train_and_test_dataset
from omegaconf import OmegaConf, DictConfig
from tools.utils import manual_seed
from models.preact_resnet import PreActResNet18
from tools.inject_backdoor import patch_trigger
import torch
import random
import PIL
from tools.dataset import get_benign_transform
import numpy as np

cam_class = HiResCAM

target_folder = '../' + 'results/imagenette/badnet/20241008013618_resnet'
path = f'{target_folder}/config.yaml'
config = OmegaConf.load(path)
# manual_seed(config.seed)
device = f'cuda:{config.device}' 
num_classes, scale = get_dataset_class_and_scale(config.dataset_name)
if config.model == "resnet18":
    net = PreActResNet18(num_classes=num_classes).to(f'cuda:{config.device}')
    target_layers = [net.layer4[-1].conv2]
elif config.model == "rnp":
    from models.resnet_cifar import resnet18
    net = resnet18(num_classes=num_classes).to(f'cuda:{config.device}')
    target_layers = [net.layer4[-1].conv2]
elif config.model == "repvgg":
    from repvgg_pytorch.repvgg import RepVGG
    net = RepVGG(num_blocks=[2, 4, 14, 1], num_classes=num_classes, width_multiplier=[1.5, 1.5, 1.5, 2.75]).to(device=f'cuda:{config.device}')
    target_layers = [net.stage4[-1].rbr_dense.conv]
    net.deploy =True
else:
    raise NotImplementedError(config.model)
ld = torch.load(f'{target_folder}/results.pth', map_location=device)
net.load_state_dict(ld['model'])
net.to(device=device)
net.eval()


_, test_dl = get_dataloader(config.dataset_name, config.batch, config.pin_memory, config.num_workers)

cam = cam_class(model=net, target_layers=target_layers)
total = 0
average_heat = None
for inputs, targets in test_dl:
    inputs, targets = inputs.to(device), targets.to(device)
    outputs = net(inputs)
    _, predicted = torch.max(outputs, 1)
    benign_heat: np.ndarray = cam(inputs, targets=[ClassifierOutputTarget(pred.item()) for pred in predicted])
    batch_average_heat = benign_heat.mean(axis=0)
    if average_heat is None:
        average_heat = batch_average_heat
    else:
        average_heat += batch_average_heat
    total += 1
average_heat /= total
x_benign = inputs[0]
y_benign = predicted[0]

from tools.dataset import PoisonDataset, get_train_and_test_dataset
from torch.utils.data.dataloader import DataLoader

_, test_ds = get_train_and_test_dataset(config.dataset_name)
config.ratio = 1
bd_test_ds = PoisonDataset(test_ds, config)
bd_test_dl = DataLoader(bd_test_ds, config.batch, False, num_workers=config.num_workers, drop_last=False)


cam = cam_class(model=net, target_layers=target_layers)
total = 0
bd_average_heat = None
for inputs, targets in bd_test_dl:
    inputs, targets = inputs.to(device), targets.to(device)
    outputs = net(inputs)
    _, predicted = torch.max(outputs, 1)
    benign_heat: np.ndarray = cam(inputs, targets=[ClassifierOutputTarget(pred.item()) for pred in predicted])
    batch_average_heat = benign_heat.mean(axis=0)
    if bd_average_heat is None:
        bd_average_heat = batch_average_heat
    else:
        bd_average_heat += batch_average_heat
    total += 1
bd_average_heat /= total
x_poison = inputs[0]
y_poison = predicted[0]


visualization_0 = show_cam_on_image(tensor2ndarray(get_de_normalization(config.dataset_name)(x_benign).squeeze()) / 255., average_heat, use_rgb=True)
visualization_1 = show_cam_on_image(tensor2ndarray(x_poison) / 255., bd_average_heat, use_rgb=True)

_, axs = plt.subplots(2, 2, figsize=(10, 8))
axs[0, 0].imshow(tensor2ndarray(get_de_normalization(config.dataset_name)(x_benign).squeeze()))
axs[0, 0].set_title(f'x_c, label: {y_benign}')
axs[0, 0].axis('off')

axs[0, 1].imshow(tensor2ndarray(x_poison))
axs[0, 1].set_title(f'x_p, label: {y_poison}')
axs[0, 1].axis('off')

axs[1, 0].imshow(visualization_0)
axs[1, 0].set_title('benign_heat')
axs[1, 0].axis('off')

axs[1, 1].imshow(visualization_1)
axs[1, 1].set_title('poison_heat')
axs[1, 1].axis('off')
plt.show()
print(f"Benign Predict: {y_benign}\n Poison Predict: {y_poison}")