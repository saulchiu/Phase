import sys
sys.path.append('../')
import torch
import pytorch_lightning as L
import random
from tools.img import rgb_to_yuv, yuv_to_rgb
from skimage.metrics import structural_similarity

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
    def __init__(self, model, lr, momentum, weight_decay, poison_rate, wind, target_label):
        super().__init__()
        self.model = model
        self.lr = lr
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.validation_step_outputs = []
        self.cur_val_loss = 0.
        self.cur_val_acc = 0.
        self.poison_rate = poison_rate
        self.wind = wind
        # self.trigger = torch.randn(size=(wind, wind), requires_grad=True)
        self.trigger = torch.nn.Parameter(torch.randn(size=(wind, wind), device=self.device, requires_grad=True))
        self.target_label = target_label
        self.save_hyperparameters()
        self.automatic_optimization = False

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch):
        opt_1, opt_2 = self.optimizers()
        x, y = batch
        if self.current_epoch % 3 == 0:
            for i in range(x.shape[0]):
                if random.random() < self.poison_rate:
                    # craft poison data
                    x_p = x[i]
                    x_p *= 255.
                    x_yuv = torch.stack(rgb_to_yuv(x_p[0], x_p[1], x_p[2]), dim=0)
                    x_u = x_yuv[1]
                    x_u_fft = torch.fft.fft2(x_u)
                    x_u_fft_imag = torch.imag(x_u_fft)
                    x_u_fft_imag[0:self.wind, 0:self.wind] = self.trigger
                    x_u_fft = torch.real(x_u_fft) + 1j * x_u_fft_imag
                    x_u = torch.real(torch.fft.ifft2(x_u_fft))
                    x_yuv[1] = x_u
                    x_p = torch.stack(yuv_to_rgb(x_yuv[0], x_yuv[1], x_yuv[2]), dim=0)
                    x_p = torch.clip(x_p, 0, 255)
                    x_p /= 255.
                    x[i] = x_p
                    y[i] = self.target_label
        y_p = self.forward(x)
        loss = torch.nn.functional.cross_entropy(y_p, y)
        opt_2.zero_grad()
        self.manual_backward(loss, retain_graph=True)  # 第一次反向传播时保留计算图
        opt_2.step()

        opt_1.zero_grad()
        self.manual_backward(loss)  # 第二次反向传播，不需要保留计算图
        opt_1.step()
        return
    
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
        trigger_optimizer = torch.optim.SGD([self.trigger], lr=self.lr, momentum=self.momentum, weight_decay=self.weight_decay)
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