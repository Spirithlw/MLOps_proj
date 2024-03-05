import typing as tp

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.optim as optim
import torchmetrics
from omegaconf import DictConfig


class CifarCNN(pl.LightningModule):
    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.cfg = cfg
        num_classes = cfg.model.num_classes

        self.block1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )
        self.block3 = torch.nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=2048, out_features=512),
            nn.Dropout1d(0.2),
            nn.GELU(),
            nn.Linear(in_features=512, out_features=128),
            nn.BatchNorm1d(128),
            nn.GELU(),
            nn.Linear(in_features=128, out_features=num_classes),
        )

        self.criterion = nn.CrossEntropyLoss()
        self.acc = torchmetrics.classification.Accuracy(
            task="multiclass", num_classes=num_classes
        )
        self.train_acc = torchmetrics.classification.Accuracy(
            task="multiclass", num_classes=num_classes
        )

    def forward(self, x):
        out = self.block2(self.block1(x))
        out = self.block3(out)
        return out

    def training_step(self, batch, batch_idx: int, dataloader_idx: int = 0):
        images, labels = batch
        preds = self(images)
        loss = self.criterion(preds, labels)
        self.train_acc(preds, labels)
        self.log_dict(
            {"train_loss": loss, "train_acc": self.train_acc},
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )

        return {"loss": loss, "train_acc": self.train_acc}

    def test_step(self, batch: tp.Any, batch_idx: int, dataloader_idx: int = 0):
        images, labels = batch
        preds = self(images)
        self.acc.update(preds, labels)

        return {"test_acc": self.acc}

    def on_test_epoch_end(self):
        self.log("test_acc", self.acc.compute())
        self.acc.reset()

    def predict_step(self, batch, batch_idx: int, dataloader_idx: int = 0) -> tp.Any:
        images, labels = batch
        outputs = self(images)
        return torch.max(outputs, 1)[1]

    def configure_optimizers(self) -> tp.Any:
        optimizer = optim.Adam(self.parameters(), lr=self.cfg.train.learning_rate)
        return optimizer

    def on_before_optimizer_step(self, optimizer):
        self.log_dict(pl.utilities.grad_norm(self, norm_type=2))
        super().on_before_optimizer_step(optimizer)
