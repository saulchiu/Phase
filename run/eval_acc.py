import sys
sys.path.append('../')
from tools.utils import manual_seed
from omegaconf import OmegaConf
import torch
from models.preact_resnet import PreActResNet18
from tools.dataset import get_dataloader, get_dataset_class_and_scale
import torch.nn.functional as F

target_folder = '../' + 'results/cifar10/inba/20241009095840_wind8'
path = f'{target_folder}/config.yaml'
config = OmegaConf.load(path)
manual_seed(config.seed)
device = f'cuda:{config.device}'
num_class, _ = get_dataset_class_and_scale(config.dataset_name)
if config.model == "resnet18":
    net = PreActResNet18(num_class)
elif config.model == "rnp":
    from models.resnet_cifar import resnet18
    net = resnet18(num_classes=num_class).to(f'cuda:{config.device}')
elif config.model == "repvgg":
    from repvgg_pytorch.repvgg import RepVGG
    net = RepVGG(num_blocks=[2, 4, 14, 1], num_classes=num_class, width_multiplier=[1.5, 1.5, 1.5, 2.75]).to(device=f'cuda:{config.device}')
    net.deploy = True
else:
    raise NotImplementedError(config.model)
ld = torch.load(f'{target_folder}/results.pth', map_location=device)
net.load_state_dict(ld['model'])
train_dl, test_dl = get_dataloader(config.dataset_name, config.batch, config.pin_memory, config.num_workers)
net.to(device)

correct = 0
total = 0
net.eval()
with torch.no_grad():
    for inputs, targets in test_dl:
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = net(inputs)
        _, predicted = torch.max(outputs, 1)
        total += targets.size(0)
        correct += (predicted == targets).sum().item()
accuracy = 100 * correct / total
print(f'Benign ACC: {accuracy:.2f}%')

from tools.dataset import PoisonDataset, get_train_and_test_dataset
from torch.utils.data.dataloader import DataLoader

_, test_ds = get_train_and_test_dataset(config.dataset_name)
config.ratio = 1
bd_test_ds = PoisonDataset(test_ds, config)
bd_test_dl = DataLoader(bd_test_ds, config.batch, False, num_workers=config.num_workers, drop_last=False)

correct = 0
total = 0
with torch.no_grad():
    for inputs, targets in bd_test_dl:
        # from tools.dataset import get_de_normalization
        # inputs = get_de_normalization(config.dataset_name)(inputs)
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = net(inputs)
        _, predicted = torch.max(outputs, 1)
        total += targets.size(0)
        correct += (predicted == targets).sum().item()
accuracy = 100 * correct / total
print(f'{config.attack.name} ASR: {accuracy:.2f}%')



