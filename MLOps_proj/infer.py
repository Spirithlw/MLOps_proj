import hydra
import numpy as np
import pytorch_lightning as pl
from dataclass import CifarDataModule
from models import CifarCNN
from omegaconf import DictConfig


@hydra.main(version_base="1.3", config_path="../conf", config_name="config")
def main(cfg: DictConfig):
    datamodule = CifarDataModule(
        data_path=cfg.data.path, batch_size=cfg.data.batch_size
    )
    model = CifarCNN.load_from_checkpoint("./CifarCNN.ckpt", cfg=cfg)

    trainer = pl.Trainer(accelerator="cpu")

    trainer.test(model, datamodule=datamodule)

    predictions = trainer.predict(model, datamodule=datamodule)
    np.savetxt("predicted_labels.csv", np.array(predictions).flatten(), delimiter=",")


if __name__ == "__main__":
    main()
