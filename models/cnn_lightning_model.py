import torch
import pytorch_lightning as L

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