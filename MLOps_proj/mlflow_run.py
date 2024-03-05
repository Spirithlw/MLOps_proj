import hydra
import mlflow
import onnx
import torch
from models import CifarCNN
from omegaconf import DictConfig


@hydra.main(version_base="1.3", config_path="../conf", config_name="config")
def main(cfg: DictConfig):
    mlflow.set_tracking_uri(cfg.mlflow.uri)
    mlflow.set_experiment(experiment_name=cfg.mlflow.experiment_name)

    onnx_model = onnx.load_model("./model.onnx")
    X = torch.randn(cfg.data.batch_size, 3, 32, 32)

    with mlflow.start_run():
        signature = mlflow.models.infer_signature(
            X.numpy(), CifarCNN(cfg)(X).detach().numpy()
        )
        model_info = mlflow.onnx.log_model(onnx_model, "model", signature=signature)

    onnx_pyfunc = mlflow.pyfunc.load_model(model_info.model_uri)
    predictions = onnx_pyfunc.predict(X.numpy())
    return predictions


if __name__ == "__main__":
    main()
