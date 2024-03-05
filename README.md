# MLOps_proj

Заходим в корень репозитория

делаем
```poetry install```

```dvc pull```

```mlflow server --host 127.0.0.1 --port 8080```

хранилище с данными на гугл диске [link](https://drive.google.com/drive/u/1/folders/1kfDazJmZWPnV-W_v_c9ZePj_4ElyV5tx)


обучение
```poetry run python3 MLOps_proj/train.py```

инфер
```poetry run python3 MLOps_proj/infer.py```

mlflow tracking server

```poetry run python3 MLOps_proj/mlflow_run.py```
