import sys
sys.path.append('../')
import PIL.Image
from torchvision.models import resnet50
from tools.dataset import get_dataloader
from tools.img import tensor2ndarray
import matplotlib.pyplot as plt
from tools.dataset import get_de_normalization, get_dataset_class_and_scale, get_train_and_test_dataset
from omegaconf import OmegaConf, DictConfig
from tools.utils import manual_seed
from classifier_models.preact_resnet import PreActResNet18
from tools.inject_backdoor import patch_trigger
import torch
import random
import PIL
from tools.dataset import get_benign_transform
import numpy as np
from tools.inject_backdoor import patch_trigger
from torchvision.io.image import read_image
from torchvision.transforms.functional import normalize, resize, to_pil_image
from torchvision.models import resnet18
from torchcam.methods import SmoothGradCAMpp
from torchcam.utils import overlay_mask

cam_class = SmoothGradCAMpp

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
    from classifier_models.resnet_cifar import resnet18
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

target_class = 9


_, test_dl = get_dataloader(config.dataset_name, config.batch, config.pin_memory, config.num_workers)
x_c = None

for batch, label in test_dl:
    batch = batch.to(device)
    for i in range(batch.shape[0]):
        if label[i] == target_class:
            x_c = batch[i]
            break
    if x_c is None:
        continue

net.load_state_dict(ld['model'])
net.to(device=device)
net.eval()
cam = cam_class(model=net, target_layer=target_layers)
out = net(x_c.unsqueeze(0))
_, y_c = torch.max(out, 1)
y_c = y_c.item()
print(y_c)
print(out)
activation_map = cam(y_c, out)
result_c = overlay_mask(to_pil_image(x_c), to_pil_image(activation_map[0].squeeze(0), mode='F'), alpha=0.5)
x_c = get_de_normalization(config.dataset_name)(x_c).squeeze()


num_classes, scale = get_dataset_class_and_scale(config.dataset_name)
if config.model == "resnet18":
    net = PreActResNet18(num_classes=num_classes).to(f'cuda:{config.device}')
    target_layers = [net.layer4[-1].conv2]
elif config.model == "rnp":
    from classifier_models.resnet_cifar import resnet18
    net = resnet18(num_classes=num_classes).to(f'cuda:{config.device}')
    target_layers = [net.layer4[-1].conv2]
elif config.model == "repvgg":
    from repvgg_pytorch.repvgg import RepVGG
    net = RepVGG(num_blocks=[2, 4, 14, 1], num_classes=num_classes, width_multiplier=[1.5, 1.5, 1.5, 2.75]).to(device=f'cuda:{config.device}')
    target_layers = [net.stage4[-1].rbr_dense.conv]
    net.deploy =True
else:
    raise NotImplementedError(config.model)
x_p = patch_trigger(x_c, config)
net.load_state_dict(ld['model'])
net.to(device=device)
net.eval()
cam = cam_class(model=net, target_layer=target_layers)
out = net(x_p.unsqueeze(0))
_, y_p = torch.max(out, 1)
y_p = y_p.item()
bd_activation_map = cam(out.squeeze(0).argmax().item(), out)
result_p = overlay_mask(to_pil_image(x_p), to_pil_image(bd_activation_map[0].squeeze(0), mode='F'), alpha=0.5)


_, axs = plt.subplots(2, 2, figsize=(10, 8))
axs[0, 0].imshow(tensor2ndarray(x_c))
axs[0, 0].set_title(f'x_c, label: {y_c}')
axs[0, 0].axis('off')

axs[0, 1].imshow(tensor2ndarray(x_p))
axs[0, 1].set_title(f'x_p, label: {y_p}')
axs[0, 1].axis('off')

axs[1, 0].imshow(result_c)
axs[1, 0].set_title('benign_heat')
axs[1, 0].axis('off')

axs[1, 1].imshow(result_p)
axs[1, 1].set_title('poison_heat')
axs[1, 1].axis('off')
plt.show()
