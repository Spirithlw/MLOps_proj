import hydra
import pytorch_lightning as pl
import torch
from dataclass import CifarDataModule
from models import CifarCNN
from omegaconf import DictConfig


@hydra.main(version_base="1.3", config_path="../conf", config_name="config")
def main(cfg: DictConfig):
    datamodule = CifarDataModule(
        data_path=cfg.data.path, batch_size=cfg.data.batch_size
    )
    model = CifarCNN(cfg)

    loggers = [
        pl.loggers.MLFlowLogger(
            experiment_name=cfg.mlflow.experiment_name, tracking_uri=cfg.mlflow.uri
        )
    ]

    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        dirpath="./", filename="CifarCNN"
    )

    trainer = pl.Trainer(
        accelerator=cfg.train.accelerator,
        max_epochs=cfg.train.epochs,
        logger=loggers,
        callbacks=[checkpoint_callback],
    )

    trainer.fit(model, datamodule=datamodule)

    batch = torch.randn(cfg.data.batch_size, 3, 32, 32)
    torch.onnx.export(model, batch, "./model.onnx")


if __name__ == "__main__":
    main()
