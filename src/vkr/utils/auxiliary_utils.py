import os
import wandb
import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, r2_score
from typing import Optional


class MetricsLogger:
    def __init__(self, save_dir: str = "experiments", name: str='model', wandb_project: Optional[str] = None):
        """
        Инициализация логгера метрик
        
        Args:
            save_dir: Папка для сохранения CSV
            wandb_project: Имя проекта Weights & Biases (если None, логирование в wandb не производится)
        """
        self.save_dir = save_dir
        self.wandb_project = wandb_project
        self.list_r2 = []
        self.list_mae = []
        self.list_iter = []
        
        # Создаем папку для сохранения
        os.makedirs(self.save_dir, exist_ok=True)
        
        # Инициализируем wandb если указан проект
        if self.wandb_project:
            wandb.init(project=self.wandb_project,
                       name=name)

    def log_metrics(self, y_true, y_pred, name_csv, iteration: int):
        """
        Логирует метрики и сохраняет их
        
        Args:
            y_true: Истинные значения
            y_pred: Предсказанные значения
            iteration: Номер итерации
        """
        # Вычисляем метрики
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        
        # Сохраняем в списки
        self.list_r2.append(r2)
        self.list_mae.append(mae)
        self.list_iter.append(iteration)
        
        # Сохраняем в CSV
        self._save_to_csv(name_csv)
        
        # Логируем в wandb если инициализирован
        if self.wandb_project and wandb.run is not None:
            wandb.log({
                "iteration": iteration,
                "MAE": mae,
                "R2": r2
            })
        
        return mae, r2

    def _save_to_csv(self, name):
        """Сохраняет метрики в CSV файл"""
        metrics_df = pd.DataFrame({
            'iteration': self.list_iter,
            'MAE': self.list_mae,
            'R2': self.list_r2
        })
        
        csv_path = os.path.join(self.save_dir, name)
        metrics_df.to_csv(csv_path, index=False)

    def finalize(self):
        """Завершает логирование (для wandb)"""
        if self.wandb_project and wandb.run is not None:
            wandb.finish()


def split_x_y(path:str) -> np.ndarray: 
    data = pd.read_csv(path)
    train_X = data.iloc[:,:-1]
    train_y = data.iloc[:,-1]
    return train_X, train_y