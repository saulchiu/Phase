import sys
sys.path.append('../')
from tools.utils import manual_seed, get_model
from omegaconf import OmegaConf
import torch
from tools.dataset import get_dataloader, get_dataset_class_and_scale, clip_normalized_tensor, get_dataset_normalization, get_de_normalization
import torch.nn.functional as F
# from tools.inject_backdoor import patch_trigger

def cal_acc_asr(target_folder):
    # target_folder = '../' + target_folder
    path = f'{target_folder}/config.yaml'
    config = OmegaConf.load(path)
    config.attack.mode = "eval"
    manual_seed(config.seed)
    device = f'cuda:{config.device}'
    num_class, _ = get_dataset_class_and_scale(config.dataset_name)
    net = get_model(config.model, num_class, device=device)
    ld = torch.load(f'{target_folder}/results.pth', map_location=device)
    net.load_state_dict(ld['model'])
    train_dl, test_dl = get_dataloader(config.dataset_name, config.batch, config.pin_memory, config.num_workers)
    net.to(device)

    correct = 0
    total = 0
    b_acc = 0
    net.eval()
    with torch.no_grad():
        for inputs, targets in test_dl:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            _, predicted = torch.max(outputs, 1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()
    accuracy = 100 * correct / total
    print(f'Benign ACC: {accuracy:.4f}%')
    b_acc = accuracy

    if config.attack.name == "benign":
        return accuracy, "-"
    
    correct = 0
    total = 0
    p_acc = 0
    do_norm = get_dataset_normalization(config.dataset_name)
    de_norm = get_de_normalization(config.dataset_name)
    sys.path.append('./run')
    sys.path.append(target_folder)
    from inject_backdoor import patch_trigger
    with torch.no_grad():
        for inputs, targets in test_dl:
            inputs, targets = inputs.to(device), targets.to(device)
            inputs = de_norm(inputs)
            bd_inpus = []
            for i in range(inputs.shape[0]):
                p = patch_trigger(inputs[i], config)
                # p = inputs[i]
                bd_inpus.append(p)
            bd_inpus = torch.stack(bd_inpus, dim=0)
            bd_inpus.clip_(0, 1)
            bd_inpus = do_norm(bd_inpus)
            targets = targets - targets + config.target_label
            outputs = net(bd_inpus)
            _, predicted = torch.max(outputs, 1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()
    accuracy = 100 * correct / total
    p_acc = accuracy
    print(f'{config.attack.name} ASR: {accuracy:.4f}%')
    return b_acc, p_acc



if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Process some paths.')
    parser.add_argument('--path', type=str, help='The path to the target folder.')
    args = parser.parse_args()
    target_folder = args.path
    cal_acc_asr(target_folder)



