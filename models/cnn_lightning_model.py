import sys
sys.path.append('../')
import torch
import pytorch_lightning as L
import random
from tools.img import rgb_to_yuv, yuv_to_rgb
from skimage.metrics import structural_similarity
from tools.dataset import get_de_normalization

class MyLightningModule(L.LightningModule):
    def __init__(self, model, lr, momentum, weight_decay):
        super().__init__()
        self.model = model
        self.lr = lr
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.validation_step_outputs = []
        self.cur_val_loss = 0.
        self. cur_val_acc = 0.
        self.save_hyperparameters()

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch):
        x, y = batch
        y_p = self.forward(x)
        loss = torch.nn.functional.cross_entropy(y_p, y)
        return loss
    
    def validation_step(self, batch):
        x, y = batch
        y_p = self.forward(x)
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
        x, y = batch
        y_p = self.forward(x)
        loss = torch.nn.functional.cross_entropy(y_p, y)
        pred_labels = torch.argmax(y_p, dim=1)
        correct = (pred_labels == y).sum().item()
        accuracy = correct / x.shape[0]
        self.log('test_loss', loss, prog_bar=True)
        self.log('test_accuracy', accuracy, prog_bar=True)
        return {"test_loss": loss, "test_accuracy": accuracy}

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.model.parameters(), lr=self.lr, momentum=self.momentum, weight_decay=self.weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)
        return {
            'optimizer': optimizer,
            "lr_scheduler": {
                        "scheduler": scheduler,
                        "monitor": self.cur_val_acc,
                        "frequency": 5,
                    },
        }


class INBALightningModule(L.LightningModule):
    def __init__(self, model, config):
        super().__init__()
        self.config = config
        self.model = model
        self.lr = config.lr
        self.momentum = config.momentum
        self.weight_decay = config.weight_decay
        self.poison_rate = config.ratio
        self.wind = config.attack.wind
        # self.trigger = torch.randn(size=(self.wind, self.wind), requires_grad=True)
        self.trigger = torch.nn.Parameter(self.init_trigger())
        self.target_label = config.target_label
        self.dataset_name = config.dataset_name
        self.save_hyperparameters(config)

        self.automatic_optimization = False
        self.validation_step_outputs = []
        self.cur_val_loss = 0.
        self.cur_val_acc = 0.
    
    def init_trigger(self):
        tg_space = torch.randn((self.wind, self.wind), device=self.device)
        tg_fft = torch.fft.fft2(tg_space)
        tg_fft_imag = torch.imag(tg_fft)
        tg_fft_imag.requires_grad_(True)
        return tg_fft_imag

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch):
        param_opt, trigger_opt = self.optimizers()
        x, y = batch
        x_list = []
        for i in range(x.shape[0]):
            if random.random() < self.poison_rate:
                # craft poison data
                x[i] = get_de_normalization(self.dataset_name)(x[i])
                x_p = x[i]
                tg_size = self.wind
                # tg_pos = random.randint(0, tg_size)
                tg_pos = 0
                x_p *= 255.
                x_yuv = torch.stack(rgb_to_yuv(x_p[0], x_p[1], x_p[2]), dim=0)
                x_yuv = torch.clip(x_yuv, 0, 255)
                x_u = x_yuv[1]
                x_u_fft = torch.fft.fft2(x_u)
                x_u_fft_imag = torch.imag(x_u_fft)
                x_u_fft_imag[tg_pos:(tg_pos+tg_size), tg_pos:(tg_pos+tg_size)] = self.trigger
                x_u_fft = torch.real(x_u_fft) + 1j * x_u_fft_imag
                x_u = torch.real(torch.fft.ifft2(x_u_fft))
                x_yuv[1] = x_u
                x_p = torch.stack(yuv_to_rgb(x_yuv[0], x_yuv[1], x_yuv[2]), dim=0)
                x_p = torch.clip(x_p, 0, 255)
                x_p /= 255.                    
                x[i] = x_p
                y[i] = self.target_label
                x_list.append(x[i])

        if len(x_list) > 0 and self.current_epoch < int(self.config / 2):
            x_list = torch.stack(x_list, dim=0)
            y_list = (torch.zeros(size=(x_list.shape[0],), device=x_list.device) + self.target_label).long()
            loss_poison = torch.nn.functional.cross_entropy(self.forward(x_list), y_list)
            trigger_opt.zero_grad()
            self.manual_backward(loss_poison, retain_graph=True)
            trigger_opt.step()

        y_p = self.forward(x)
        loss = torch.nn.functional.cross_entropy(y_p, y)
        param_opt.zero_grad()
        self.manual_backward(loss)
        param_opt.step()
        # return loss
    
    def validation_step(self, batch):
        x, y = batch
        y_p = self.forward(x)
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
        x, y = batch
        y_p = self.forward(x)
        loss = torch.nn.functional.cross_entropy(y_p, y)
        pred_labels = torch.argmax(y_p, dim=1)
        correct = (pred_labels == y).sum().item()
        accuracy = correct / x.shape[0]
        self.log('test_loss', loss, prog_bar=True)
        self.log('test_accuracy', accuracy, prog_bar=True)
        return {"test_loss": loss, "test_accuracy": accuracy}

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.model.parameters(), lr=self.lr, momentum=self.momentum, weight_decay=self.weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)
        trigger_optimizer = torch.optim.SGD([self.trigger], lr=self.lr * 10, momentum=self.momentum, weight_decay=self.weight_decay)
        return (
            {
                'optimizer': optimizer,
                "lr_scheduler": {
                        "scheduler": scheduler,
                        "monitor": self.cur_val_acc,
                        "frequency": 5,
                        },
            },
            {
                'optimizer': trigger_optimizer
            }
        )