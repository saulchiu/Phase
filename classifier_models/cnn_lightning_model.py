REPO_ROOT='/home/chengyiqiu/code/INBA/'
import sys
sys.path.append(REPO_ROOT)
import torch
import pytorch_lightning as L
import random
from tools.img import rgb_to_yuv, yuv_to_rgb
from skimage.metrics import structural_similarity
from tools.dataset import get_de_normalization
from ema_pytorch.ema_pytorch import EMA
from torchmetrics.image import StructuralSimilarityIndexMeasure
import matplotlib.pyplot as plt
import os
from tools.dataset import get_dataset_class_and_scale
import math
from scipy import stats

def get_optimizer(model, learning_rate, weight_decay):
    param_dict = {pn: p for pn, p in model.named_parameters()}
    parameters_decay, parameters_no_decay = model.separate_parameters()
    
    optim_groups = [
        {"params": [param_dict[pn] for pn in parameters_decay], "weight_decay": weight_decay},
        {"params": [param_dict[pn] for pn in parameters_no_decay], "weight_decay": 0.0},
    ]
    optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate)
    return optimizer

def visualize_metrics(metrics_list, target_folder):
    epochs = [m['epoch'] for m in metrics_list]
    train_loss = [m['train_loss_epoch'].cpu().item() if isinstance(m['train_loss_epoch'], torch.Tensor) else m['train_loss_epoch'] for m in metrics_list]
    train_acc = [m['train_acc_epoch'].cpu().item() if isinstance(m['train_acc_epoch'], torch.Tensor) else m['train_acc_epoch'] for m in metrics_list]
    val_loss = [m['val_loss_epoch'].cpu().item() if isinstance(m['val_loss_epoch'], torch.Tensor) else m['val_loss_epoch'] for m in metrics_list]
    val_acc = [m['val_acc_epoch'].cpu().item() if isinstance(m['val_acc_epoch'], torch.Tensor) else m['val_acc_epoch'] for m in metrics_list]
    fig, ax1 = plt.subplots()
    ax1.plot(epochs, train_loss, label='Train Loss', color='blue', linestyle='-')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.plot(epochs, val_loss, label='Val Loss', color='blue', linestyle='--')
    ax1.legend(loc='upper left')
    ax2 = ax1.twinx()
    ax2.set_ylabel('Accuracy')
    ax2.plot(epochs, train_acc, label='Train Accuracy', color='green', linestyle='-')
    ax2.plot(epochs, val_acc, label='Val Accuracy', color='green', linestyle='--')
    ax2.legend(loc='upper right')
    plt.savefig(f"{target_folder}/metrics_plot.png")
    plt.close()

class BASELightningModule(L.LightningModule):
    def __init__(self, model, config):
        super().__init__()
        self.config = config
        self.model = model
        # self.ema = EMA(self.model, update_every=self.config.ema_update_every)
        # self.ema.to(device=self.device)
        self.metrics_list = []
        self.cur_val_loss = 0.
        self. cur_val_acc = 0.
        self.automatic_optimization = False
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.config.amp)
        if config.model != "convnext":
            self.opt = torch.optim.SGD(filter(lambda p: p.requires_grad, self.model.parameters()),
                                        lr=self.config.lr, 
                                        momentum=self.config.momentum, 
                                        weight_decay=self.config.weight_decay)
        else:
            print("-" * 10)
            print('use COnvNext opt')
            print('-' * 10)
            self.opt=get_optimizer(model, learning_rate=config.lr, weight_decay=config.weight_decay)
        self.schedule = torch.optim.lr_scheduler.CosineAnnealingLR(self.opt, T_max=config.epoch)
        self.criterion = torch.nn.CrossEntropyLoss()

    def forward(self, x):
        return self.model(x)

    def on_train_epoch_end(self):
        self.schedule.step()
        metrics = {
            "epoch": self.current_epoch,
            "train_loss_epoch": self.trainer.logged_metrics.get('train_loss', None),
            "train_acc_epoch": self.trainer.logged_metrics.get('train_acc', None)
        }
        if 'val_loss' in self.trainer.logged_metrics:
            metrics["val_loss_epoch"] = self.trainer.logged_metrics['val_loss'].item()
        else:
            metrics["val_loss_epoch"] = None

        if 'val_acc' in self.trainer.logged_metrics:
            metrics["val_acc_epoch"] = self.trainer.logged_metrics['val_acc'].item()
        else:
            metrics["val_acc_epoch"] = None
        self.metrics_list.append(metrics)

    def training_step(self, batch):
        self.model.train()
        self.model.to(self.device, non_blocking=self.config.non_blocking)
        x, y = batch
        x = x.to(self.device, non_blocking=self.config.non_blocking)
        y = y.to(self.device, non_blocking=self.config.non_blocking)
        with torch.amp.autocast('cuda', enabled=self.config.amp):
            y_p = self.forward(x)
            loss = self.criterion(y_p, y.long())
        _, predicted = torch.max(y_p, -1)
        correct = predicted.eq(y).sum().item()
        total = y.size(0)
        self.scaler.scale(loss).backward()
        self.scaler.step(self.opt)
        self.scaler.update()
        self.opt.zero_grad()
        self.log('train_loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log('train_acc', correct / total, on_step=False, on_epoch=True, prog_bar=True, logger=True)
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        x = x.to(self.device, non_blocking=self.config.non_blocking)
        y = y.to(self.device, non_blocking=self.config.non_blocking)
        y_p = self.forward(x)
        loss = self.criterion(y_p, y.long())
        _, predicted = torch.max(y_p, -1)
        correct = predicted.eq(y).sum().item()
        total = y.size(0)
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log('val_acc', correct / total, on_step=False, on_epoch=True, prog_bar=True, logger=True)


    def test_step(self, batch, batch_idx):
        x, y = batch
        x = x.to(self.device, non_blocking=self.config.non_blocking)
        y = y.to(self.device, non_blocking=self.config.non_blocking)
        y_p = self.forward(x)
        loss = self.criterion(y_p, y.long())
        _, predicted = torch.max(y_p, -1)
        correct = predicted.eq(y).sum().item()
        total = y.size(0)
        self.log('test_loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log('test_acc', correct / total, on_step=False, on_epoch=True, prog_bar=True, logger=True)

    def configure_optimizers(self):
        # optimizer = torch.optim.SGD(self.model.parameters(), lr=self.lr, momentum=self.momentum, weight_decay=self.weight_decay)
        # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)
        # return {
        #     'optimizer': optimizer,
        #     "lr_scheduler": {
        #                 "scheduler": scheduler,
        #                 "monitor": self.cur_val_acc,
        #                 "frequency": 1,
        #             },
        # }
        return None




class INBALightningModule(L.LightningModule):
    def __init__(self, model: torch.nn.Module, config, mode):
        super().__init__()
        self.config = config
        self.mode = mode
        self.model = model
        self.lr = config.lr
        self.momentum = config.momentum
        self.weight_decay = config.weight_decay
        self.poison_rate = config.ratio
        _, scale = get_dataset_class_and_scale(config.dataset_name)
        self.target_label = config.target_label
        self.dataset_name = config.dataset_name
        self.automatic_optimization = False
        self.validation_step_outputs = []
        self.cur_val_loss = 0.
        self.cur_val_acc = 0.
        self.reset_epoch_flag = False
        self.metrics_list = []
        self.model_state_dict_backup = model.state_dict().copy()
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.config.amp)
        if mode == "trigger":  # train trigger or mask
            # raise NotImplementedError
            print('------')
            print('train trigger')
            print('------')
            tg_size = config.attack.tg_size * 2
            self.u_tg = torch.nn.Parameter(
                torch.full((tg_size, tg_size), math.pi, requires_grad=True)
            )
            tg_size = int(config.attack.tg_size * config.attack.v_size_coeff) * 2
            self.v_tg = torch.nn.Parameter(
                torch.full((tg_size, tg_size), math.pi, requires_grad=True)
            )
            self.tg_opt = torch.optim.SGD(filter(lambda p: p.requires_grad, [self.u_tg, self.v_tg]),
                                        lr=self.config.lr, 
                                        momentum=self.config.momentum, 
                                        weight_decay=self.config.weight_decay)
            self.param_opt = torch.optim.SGD(filter(lambda p: p.requires_grad, self.model.parameters()),
                                        lr=self.config.lr, 
                                        momentum=self.config.momentum, 
                                        weight_decay=self.config.weight_decay)
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.param_opt, T_max=self.config.attack.tg_epoch)
        elif mode == 'model':  # train model
            print('------')
            print('train model')
            print('------')
            self.u_tg = torch.load(f'{config.path}/trigger.pth')['u_tg']
            self.v_tg = torch.load(f'{config.path}/trigger.pth')['v_tg']
            self.param_opt = torch.optim.SGD(filter(lambda p: p.requires_grad, self.model.parameters()),
                                        lr=self.config.lr, 
                                        momentum=self.config.momentum, 
                                        weight_decay=self.config.weight_decay)
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.param_opt, T_max=self.config.epoch)
        self.criterion = torch.nn.CrossEntropyLoss()

    def forward(self, x):
        return self.model(x)
    
    def on_train_epoch_end(self):
        self.scheduler.step()
        if self.mode == "trigger":
            if self.trainer.current_epoch == self.config.attack.tg_epoch - 1:
                tg = {
                    "u_tg": torch.clamp(self.u_tg.data.detach(), -math.pi, math.pi),
                    "v_tg": torch.clamp(self.v_tg.data.detach(), -math.pi, math.pi)
                }
                print('-' * 10)
                print(tg['u_tg'][:,0])
                print('-' * 10)
                torch.save(tg, f'{self.config.path}/trigger.pth')
            # raise NotImplementedError
            return
        if self.mode == "model":
            metrics = {
                "epoch": self.current_epoch,
                "train_loss_epoch": self.trainer.logged_metrics.get('train_loss', None),
                "train_acc_epoch": self.trainer.logged_metrics.get('train_acc', None)
            }
            if 'val_loss' in self.trainer.logged_metrics:
                metrics["val_loss_epoch"] = self.trainer.logged_metrics['val_loss'].item()
            else:
                metrics["val_loss_epoch"] = None

            if 'val_acc' in self.trainer.logged_metrics:
                metrics["val_acc_epoch"] = self.trainer.logged_metrics['val_acc'].item()
            else:
                metrics["val_acc_epoch"] = None
            self.metrics_list.append(metrics)


    def training_step(self, batch):
        # torch.autograd.set_detect_anomaly(True)
        self.model.train()
        # self.ema.train()
        x, y = batch
        x_list = []
        x_p_list = []
        ssim_function = StructuralSimilarityIndexMeasure(data_range=1.0).to(self.device)
        for i in range(x.shape[0]):
            if random.random() < self.poison_rate:
                # craft poison data
                x_p = x[i].clone()
                x_yuv = torch.stack(rgb_to_yuv(x_p[0], x_p[1], x_p[2]), dim=0)
                
                # Y channel
                # x_y = x_yuv[0]
                # x_y_fft = torch.fft.fft2(x_y)
                # x_y_fft_real = torch.real(x_y_fft)
                # x_y_fft_imag = torch.imag(x_y_fft) * self.mask.clone()
                # x_y_fft = x_y_fft_real + 1j * x_y_fft_imag
                # x_y = torch.real(torch.fft.ifft2(x_y_fft))
                # x_yuv[0] = x_y

                # U channel
                scale = x.shape[-1]
                tg_pos = int(scale / 2)
                tg_size = self.config.attack.tg_size
                x_u = x_yuv[1]
                x_u_fft = torch.fft.fft2(x_u)
                x_u_fft_amp = torch.abs(x_u_fft)
                x_u_fft_pha = torch.angle(x_u_fft)
                x_u_fft_pha[tg_pos-tg_size:tg_pos+tg_size, tg_pos-tg_size:tg_pos+tg_size] = self.u_tg
                x_u_fft = x_u_fft_amp * torch.exp(1j * x_u_fft_pha)
                x_u = torch.fft.ifft2(x_u_fft)
                x_u = torch.real(x_u)
                x_yuv[1] = x_u

                # V channel
                tg_size = int(tg_size * self.config.attack.v_size_coeff)
                x_v = x_yuv[2]
                x_v_fft = torch.fft.fft2(x_v)
                x_v_fft_amp = torch.abs(x_v_fft)
                x_v_fft_pha = torch.angle(x_v_fft)
                x_v_fft_pha[tg_pos-tg_size:tg_pos+tg_size, tg_pos-tg_size:tg_pos+tg_size] = self.v_tg
                x_v_fft = x_v_fft_amp * torch.exp(1j * x_v_fft_pha)
                x_v = torch.fft.ifft2(x_v_fft)
                x_v = torch.real(x_v)
                x_yuv[2] = x_v
                
                x_p = torch.stack(yuv_to_rgb(x_yuv[0], x_yuv[1], x_yuv[2]), dim=0)

                # mix amp
                x_c = x[i].clone()
                x_c_fft = torch.fft.fft2(x_c, dim=(1, 2))
                x_p_fft = torch.fft.fft2(x_p, dim=(1, 2))
                x_p_fft = torch.abs(x_c_fft) * torch.exp(1j * torch.angle(x_p_fft))
                x_p = torch.fft.ifft2(x_p_fft, dim=(1, 2))
                x_p = torch.real(x_p)

                if self.mode == "trigger":
                    x_p_list.append(x_p)
                    x_list.append(x[i].clone())
                x[i] = x_p
                y[i] = self.target_label

        if len(x_p_list) > 1:  # x_p.shape[0] should > 1
            x_p_list = torch.stack(x_p_list, dim=0)
            x_list = torch.stack(x_list, dim=0)
            ssim_tensor = ssim_function(x_p_list, x_list)
            y_list = (torch.zeros(size=(x_p_list.shape[0],), device=x_p_list.device) + self.target_label).long()
            loss_poison = torch.nn.functional.cross_entropy(self.forward(x_p_list), y_list)
            loss_poison = loss_poison + (1. - ssim_tensor) * self.config.attack.ssim_coeff
            self.manual_backward(loss_poison, retain_graph=True)
            self.tg_opt.step()
            self.tg_opt.zero_grad()
        y_p = self.forward(x)
        loss = self.criterion(y_p, y.long())
        _, predicted = torch.max(y_p, -1)
        correct = predicted.eq(y).sum().item()
        total = y.size(0)
        self.scaler.scale(loss).backward()
        self.scaler.step(self.param_opt)
        self.scaler.update()
        self.param_opt.zero_grad()
        self.log('train_loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log('train_acc', correct / total, on_step=False, on_epoch=True, prog_bar=True, logger=True)
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        x = x.to(self.device, non_blocking=self.config.non_blocking)
        y = y.to(self.device, non_blocking=self.config.non_blocking)
        y_p = self.forward(x)
        loss = self.criterion(y_p, y.long())
        _, predicted = torch.max(y_p, -1)
        correct = predicted.eq(y).sum().item()
        total = y.size(0)
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log('val_acc', correct / total, on_step=False, on_epoch=True, prog_bar=True, logger=True)


    def test_step(self, batch, batch_idx):
        x, y = batch
        x = x.to(self.device, non_blocking=self.config.non_blocking)
        y = y.to(self.device, non_blocking=self.config.non_blocking)
        y_p = self.forward(x)
        loss = self.criterion(y_p, y.long())
        _, predicted = torch.max(y_p, -1)
        correct = predicted.eq(y).sum().item()
        total = y.size(0)
        self.log('test_loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log('test_acc', correct / total, on_step=False, on_epoch=True, prog_bar=True, logger=True)


    def configure_optimizers(self):
        # optimizer = torch.optim.SGD(self.model.parameters(), lr=self.lr, momentum=self.momentum, weight_decay=self.weight_decay)
        # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)
        # trigger_optimizer = torch.optim.SGD([self.trigger], lr=self.lr, momentum=self.momentum, weight_decay=self.weight_decay)
        # return (
        #     {
        #         'optimizer': optimizer,
        #         "lr_scheduler": {
        #                 "scheduler": scheduler,
        #                 "monitor": self.cur_val_acc,
        #                 "frequency": 5,
        #                 },
        #     },
        #     {
        #         'optimizer': trigger_optimizer
        #     }
        # )
        return None

