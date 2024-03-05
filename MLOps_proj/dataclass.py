from typing import Optional

import pytorch_lightning as pl
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


class CifarDataModule(pl.LightningDataModule):
    def __init__(self, data_path, batch_size):
        super().__init__()
        self.save_hyperparameters()
        self.data_path = data_path
        self.batch_size = batch_size

    def setup(self, stage: Optional[str] = None):
        self.train_dataset = datasets.CIFAR10(
            root=self.data_path,
            train=True,
            download=False,
            transform=transforms.Compose([transforms.ToTensor()]),
        )

        self.test_dataset = datasets.CIFAR10(
            root=self.data_path,
            train=False,
            download=False,
            transform=transforms.Compose([transforms.ToTensor()]),
        )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset, batch_size=self.batch_size, num_workers=4, shuffle=True
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_dataset, batch_size=self.batch_size, num_workers=4, shuffle=False
        )

    def predict_dataloader(self) -> DataLoader:
        return self.test_dataloader()
