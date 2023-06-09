# AUTOGENERATED! DO NOT EDIT! File to edit: ../nbs/02_core.ipynb.

# %% auto 0
__all__ = ['SLModel']

# %% ../nbs/02_core.ipynb 2
from .utils import *
from .imports import *
from .data import *

# %% ../nbs/02_core.ipynb 4
class SLModel(L.LightningModule):
    "LightningModule for single label classification."
    
    def __init__(self, model, learning_rate, cosine_t_max, in_chans=3, seed=42):
        super().__init__()
        L.seed_everything(seed)
        store_attr("model,learning_rate,cosine_t_max")
        self.example_input_array = torch.Tensor(1, in_chans, 224, 224)
        self.num_classes = model.num_classes
        self.save_hyperparameters(ignore=["model"])
        self.train_acc = MulticlassAccuracy(average=None, num_classes=self.num_classes)
        self.val_acc = MulticlassAccuracy(average=None, num_classes=self.num_classes)
        self.test_acc = MulticlassAccuracy(average=None, num_classes=self.num_classes)
        # self.train_acc = TM.Accuracy(task="multiclass", num_classes=self.num_classes)
        # self.val_acc = TM.Accuracy(task="multiclass", num_classes=self.num_classes)
        # self.test_acc = TM.Accuracy(task="multiclass", num_classes=self.num_classes)

    def forward(self, x):
        return self.model(x)

    def _shared_step(self, batch):
        features, true_labels = batch['image'], batch['label']
        logits = self(features)
        loss = F.cross_entropy(logits, true_labels)
        predicted_labels = torch.argmax(logits, dim=1)
        return loss, true_labels, predicted_labels

    def training_step(self, batch, batch_idx):
        loss, true_labels, predicted_labels = self._shared_step(batch)
        self.log("train_loss", loss)
        self.train_acc(predicted_labels, true_labels)
        self.log("train_acc", self.train_acc, prog_bar=True, on_epoch=True, on_step=False)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, true_labels, predicted_labels = self._shared_step(batch)
        self.log("val_loss", loss, prog_bar=True)
        self.val_acc(predicted_labels, true_labels)
        self.log("val_acc", self.val_acc, prog_bar=True)

    def test_step(self, batch, batch_idx):
        loss, true_labels, predicted_labels = self._shared_step(batch)
        self.test_acc(predicted_labels, true_labels)
        self.log("test_acc", self.test_acc)

    def predict_step(self, batch, batch_idx):
        images = batch["image"]
        logits = self(images)
        predicted_labels = torch.argmax(logits, dim=1)
        return predicted_labels
    
    def configure_optimizers(self):
        opt = optim.Adam(self.parameters(), lr=self.learning_rate)
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=self.cosine_t_max)
        return {"optimizer": opt, "lr_scheduler": {"scheduler": sch, "monitor": "train_loss",
                                                   "interval": "step", "frequency": 1}}
