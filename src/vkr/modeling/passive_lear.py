import hydra
from pathlib import Path
import numpy as np
import time
import pandas as pd
from omegaconf import DictConfig, OmegaConf
from typing import Union
from sklearn.linear_model import SGDRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from joblib import dump
import sys
project_root = Path(__file__).absolute().parents[3]  # До My_VKR
sys.path.insert(0, str(project_root))
print(project_root)
from src.vkr.conf.configs import ModelConfig, BaselineTrainConfig
from src.vkr.models.classes_models import GPRegressor,SerializableKANWrapper,baseline_models
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score,mean_absolute_percentage_error
import wandb
from src.vkr.utils.auxiliary_utils import split_x_y


wandb.login()

@hydra.main(
    version_base=None, config_path="../../../configs", config_name="config"
)
def study_model(cfg : DictConfig) -> None:
    OmegaConf.to_yaml(cfg)
    train_config = BaselineTrainConfig(**cfg.get("train", {}))
    train_X, train_y = split_x_y(train_config.train_data_path)
    val_X, val_y = split_x_y(train_config.val_data_path)
    model_config = ModelConfig(**cfg.get("model", {}))
    models = baseline_models(model_config.parameters)
    print(models,'models')
    create_model_study(train_X, train_y, val_X, val_y, models, model_config.model_info_dir)



def create_model_study(train_X: np.ndarray,
    train_y: np.ndarray,
    val_X: np.ndarray,
    val_y: np.ndarray,
    models: 
        Union[
            SGDRegressor,
            GradientBoostingRegressor,
            RandomForestRegressor,
            GPRegressor,
            SerializableKANWrapper,
        ],
    model_info_dir: str,
    project_name: str = "models_passive",
    ) -> None:
    for model_name, model in models.items():
        timestamp = int(time.time())
        run_name = f"{model_name}_{timestamp}"
        
        with wandb.init(project=project_name, 
                       name=run_name,
                       reinit=True,
                       settings=wandb.Settings(symlink=False),
                       tags=["experiment", model_name, f"version_{timestamp}"]) as run:
            
            print(f"Training {model_name}...")
            
            try:
                # Обучение модели
                model.fit(train_X, train_y)
                
                # Предсказания
                train_preds = model.predict(train_X)
                val_preds = model.predict(val_X)

                # Метрики
                metrics = {
                    "train": {
                        "r2": r2_score(train_y, train_preds),
                        "mse": mean_squared_error(train_y, train_preds),
                        "mape": mean_absolute_percentage_error(train_y, train_preds),
                        "mae": mean_absolute_error(train_y, train_preds)
                    },
                    "val": {
                        "r2": r2_score(val_y, val_preds),
                        "mse": mean_squared_error(val_y, val_preds),
                        "mape": mean_absolute_percentage_error(val_y, val_preds),
                        "mae": mean_absolute_error(val_y, val_preds)
                    }
                }
                
                # Логирование метрик
                wandb.log({
                    "train/mae": metrics["train"]["mae"],
                    "train/mape": metrics["train"]["mape"],
                    "train/mse": metrics["train"]["mse"],
                    "train/r2": metrics["train"]["r2"],
                    "val/mape": metrics["val"]["mape"],
                    "val/mae": metrics["val"]["mae"],
                    "val/mse": metrics["val"]["mse"],
                    "val/r2": metrics["val"]["r2"],
                })
                
                # Сохранение модели (особый случай для KAN)
                model_dir = Path(model_info_dir) / model_name
                model_dir.mkdir(parents=True, exist_ok=True)
                
                if isinstance(model, SerializableKANWrapper):
                    # Для KAN используем специальный метод сохранения
                    model_path = model_dir / f"model_{timestamp}.pt"
                    model.save(model_path)
                    
                    # Создаем артефакт
                    artifact = wandb.Artifact(
                        name=f"{model_name}_model",
                        type="model",
                        description=f"KAN model ({model_name})"
                    )
                    artifact.add_file(str(model_path))
                    run.log_artifact(artifact)
                else:
                    # Для других моделей используем стандартный подход
                    model_path = model_dir / f"model_{timestamp}.joblib"
                    dump(model, model_path)
                    
                    artifact = wandb.Artifact(
                        name=f"{model_name}_model",
                        type="model",
                        description=f"{type(model).__name__} model"
                    )
                    artifact.add_file(str(model_path))
                    run.log_artifact(artifact)
                
                # Сохранение информации о модели
                info_path = model_dir / f"model_info_{timestamp}.json"
                model_info = {
                    "model_name": model_name,
                    "version": timestamp,
                    "metrics": metrics,
                    "model_path": str(model_path)
                }
                dump(model_info, info_path)
                
                info_artifact = wandb.Artifact(
                    name=f"{model_name}_info",
                    type="metadata",
                    description="Model metadata"
                )
                info_artifact.add_file(str(info_path))
                run.log_artifact(info_artifact)
                
            except Exception as e:
                print(f"Error with {model_name}: {str(e)}")
                wandb.log({"error": str(e)})
                continue

    return None

if __name__ == "__main__":
    study_model()
    