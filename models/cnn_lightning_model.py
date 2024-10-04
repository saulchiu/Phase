import sys
sys.path.append('../')
import torch
import pytorch_lightning as L
import random
from tools.img import rgb_to_yuv, yuv_to_rgb
from skimage.metrics import structural_similarity
from tools.dataset import get_de_normalization
from ema_pytorch.ema_pytorch import EMA
from torchmetrics.image import StructuralSimilarityIndexMeasure

class BASELightningModule(L.LightningModule):
    def __init__(self, model, config):
        super().__init__()
        self.model = model
        self.ema = EMA(self.model, update_every=10)
        self.ema.to(device=self.device)
        self.lr = config.lr
        self.momentum = config.momentum
        self.weight_decay = config.weight_decay
        self.validation_step_outputs = []
        self.cur_val_loss = 0.
        self. cur_val_acc = 0.
        self.automatic_optimization = False
        self.opt = torch.optim.SGD(self.model.parameters(), lr=self.lr, momentum=self.momentum, weight_decay=self.weight_decay)
        self.schedule = torch.optim.lr_scheduler.CosineAnnealingLR(self.opt, T_max=config.epoch)

    def forward(self, x):
        return self.model(x)

    def on_train_epoch_end(self):
        self.schedule.step()

    def training_step(self, batch):
        self.model.train()
        self.ema.train()
        x, y = batch
        y_p = self.forward(x)
        loss = torch.nn.functional.cross_entropy(y_p, y)
        self.opt.zero_grad()
        self.manual_backward(loss)
        self.opt.step()
        self.ema.update()
        # return loss
    
    def validation_step(self, batch):
        self.model.eval()
        self.ema.eval()
        x, y = batch
        y_p = self.ema(x)
        loss = torch.nn.functional.cross_entropy(y_p, y)
        pred_labels = torch.argmax(y_p, dim=1)
        correct = (pred_labels == y).sum().item()
        accuracy = correct / x.shape[0]
        self.validation_step_outputs.append(accuracy)
        self.log('val_loss', loss, prog_bar=True)
        self.log('val_accuracy', accuracy, prog_bar=True)
        self.cur_val_loss = loss
        self.cur_val_acc = accuracy
        return accuracy
    
    def test_step(self, batch):
        self.model.eval()
        self.ema.eval()
        x, y = batch
        y_p = self.ema(x)
        loss = torch.nn.functional.cross_entropy(y_p, y)
        pred_labels = torch.argmax(y_p, dim=1)
        correct = (pred_labels == y).sum().item()
        accuracy = correct / x.shape[0]
        self.log('test_loss', loss, prog_bar=True)
        self.log('test_accuracy', accuracy, prog_bar=True)
        return {"test_loss": loss, "test_accuracy": accuracy}

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
    def __init__(self, model: torch.nn.Module, config):
        super().__init__()
        self.config = config
        self.model = model
        self.ema = EMA(self.model, update_every=10)
        self.ema.to(device=self.device)
        self.lr = config.lr
        self.momentum = config.momentum
        self.weight_decay = config.weight_decay
        self.poison_rate = config.ratio
        self.trigger = torch.nn.Parameter(self.init_trigger())
        self.target_label = config.target_label
        self.dataset_name = config.dataset_name
        self.automatic_optimization = False
        self.validation_step_outputs = []
        self.cur_val_loss = 0.
        self.cur_val_acc = 0.

        self.extra_epochs  =int(self.config.epoch / 10)
        self.reset_epoch_flag = False

        self.model_state_dict_backup = model.state_dict()
        self.tg_opt = torch.optim.SGD([self.trigger], lr=self.lr, momentum=self.momentum, weight_decay=self.weight_decay)
        self.param_opt = torch.optim.SGD(self.model.parameters(), lr=self.lr, momentum=self.momentum, weight_decay=self.weight_decay)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.param_opt, T_max=self.config.epoch)
    
    def init_trigger(self):
        print('-----train trigger first-----')
        tg_spatial = torch.randn((self.config.attack.wind, self.config.attack.wind), device=self.device)
        tg_fft = torch.fft.fft2(tg_spatial)
        tg_fft_imag = torch.imag(tg_fft)
        tg_fft_imag.requires_grad_(True)
        return tg_fft_imag

    def forward(self, x):
        return self.model(x)
    
    def on_train_epoch_end(self):
        if self.extra_epochs > 0:
            self.extra_epochs -= 1
        elif self.extra_epochs == 0:
            print('-----start train model-----')
            self.model.load_state_dict(self.model_state_dict_backup)
            self.param_opt = torch.optim.SGD(self.model.parameters(), lr=self.lr, momentum=self.momentum, weight_decay=self.weight_decay)
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.param_opt, T_max=self.config.epoch)
            self.extra_epochs = -1
        self.scheduler.step()


    def training_step(self, batch):
        torch.autograd.set_detect_anomaly(True)
        self.model.train()
        self.ema.train()
        x, y = batch
        x_list = []
        x_p_list = []
        ssim_function = StructuralSimilarityIndexMeasure(data_range=1.0).to(self.device)
        for i in range(x.shape[0]):
            if random.random() < self.poison_rate:
                # craft poison data
                x[i] = get_de_normalization(self.dataset_name)(x[i]).squeeze()
                x_p = x[i] * 255.
                tg_size = self.config.attack.wind
                tg_pos = 0 if self.config.attack.rand_pos == 0 else random.randint(0, self.config.attack.wind)
                x_yuv = torch.stack(rgb_to_yuv(x_p[0], x_p[1], x_p[2]), dim=0)
                x_yuv = torch.clip(x_yuv, 0, 255)
                x_u = x_yuv[self.config.attack.target_channel]
                x_u_fft = torch.fft.fft2(x_u)
                x_u_fft_imag = torch.imag(x_u_fft)
                x_u_fft_imag[tg_pos:(tg_pos+tg_size), tg_pos:(tg_pos+tg_size)] = self.trigger
                x_u_fft = torch.real(x_u_fft) + 1j * x_u_fft_imag
                x_u = torch.real(torch.fft.ifft2(x_u_fft))
                x_yuv[self.config.attack.target_channel] = x_u
                x_p = torch.stack(yuv_to_rgb(x_yuv[0], x_yuv[1], x_yuv[2]), dim=0)
                x_p = torch.clip(x_p, 0, 255)
                x_p /= 255.
                if self.extra_epochs > 0:
                    x_p_list.append(x_p)
                    x_list.append(x[i].clone())
                x[i] = x_p
                y[i] = self.target_label

        # if poison rate is 0, never into this loop
        if len(x_p_list) > 0 and self.extra_epochs > 0:
            x_p_list = torch.stack(x_p_list, dim=0)
            x_list = torch.stack(x_list, dim=0)
            ssim_tensor = ssim_function(x_p_list, x_list)
            y_list = (torch.zeros(size=(x_p_list.shape[0],), device=x_p_list.device) + self.target_label).long()
            loss_poison = torch.nn.functional.cross_entropy(self.forward(x_p_list), y_list)
            loss_poison = loss_poison + (1. - ssim_tensor) * self.config.attack.ssim_coeff
            self.tg_opt.zero_grad()
            self.manual_backward(loss_poison, retain_graph=True)
            self.tg_opt.step()
        y_p = self.forward(x)
        loss = torch.nn.functional.cross_entropy(y_p, y)
        self.param_opt.zero_grad()
        self.manual_backward(loss)
        self.param_opt.step()
        # return loss
    
    def validation_step(self, batch):
        self.model.eval()
        self.ema.eval()
        x, y = batch
        y_p = self.ema(x)
        loss = torch.nn.functional.cross_entropy(y_p, y)
        pred_labels = torch.argmax(y_p, dim=1)
        correct = (pred_labels == y).sum().item()
        accuracy = correct / x.shape[0]
        self.validation_step_outputs.append(accuracy)
        self.log('val_loss', loss, prog_bar=True)
        self.log('val_accuracy', accuracy, prog_bar=True)
        self.cur_val_loss = loss
        self.cur_val_acc = accuracy
        return accuracy
    
    def test_step(self, batch):
        self.model.eval()
        self.ema.eval()
        x, y = batch
        y_p = self.ema(x)
        loss = torch.nn.functional.cross_entropy(y_p, y)
        pred_labels = torch.argmax(y_p, dim=1)
        correct = (pred_labels == y).sum().item()
        accuracy = correct / x.shape[0]
        self.log('test_loss', loss, prog_bar=True)
        self.log('test_accuracy', accuracy, prog_bar=True)
        return {"test_loss": loss, "test_accuracy": accuracy}

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