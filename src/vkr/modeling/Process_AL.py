import hydra
from pathlib import Path
import numpy as np
import pandas as pd
from omegaconf import DictConfig, OmegaConf
from sklearn.metrics import mean_absolute_error, r2_score
import wandb
import torch
import sys
import gpytorch
from gpytorch.kernels import ScaleKernel, RBFKernel, MaternKernel
from gpytorch.mlls import ExactMarginalLogLikelihood
from gpytorch.likelihoods import GaussianLikelihood
project_root = Path(__file__).absolute().parents[3]  # До My_VKR
sys.path.insert(0, str(project_root))
from src.vkr.conf.configs import BaselineTrainConfig, ALTrainConfig
from src.vkr.models.classes_models import *
from src.vkr.models.AL_strategies import *
from src.vkr.utils.auxiliary_utils import *


wandb.login()

@hydra.main(
    version_base=None, config_path="../../../configs", config_name="config"
)
def active_lear(cfg : DictConfig) -> None:
    OmegaConf.to_yaml(cfg)
    train_config = BaselineTrainConfig(**cfg.get("train", {}))
    models_parametrs = ALTrainConfig(**cfg.get("model", {}))
    X_pool, y_pool = split_x_y(train_config.normalized_data_path)
    np.random.seed(42)
    INIT_SIZE = models_parametrs.n_start_points
    initial_idx = np.random.choice(len(y_pool), size=INIT_SIZE, replace=False)
    X_train, y_train = X_pool.iloc[initial_idx], y_pool.iloc[initial_idx]
    X_pool, y_pool = (
        np.delete(X_pool, initial_idx, axis=0),
        np.delete(y_pool, initial_idx, axis=0),
    )
    if models_parametrs.query_strategy_type == 'qbc':
        process_qbc(X_train, y_train, models_parametrs, X_pool, y_pool)
    elif models_parametrs.query_strategy_type == 'naqbc':
        process_naqbc(X_train, y_train, models_parametrs, X_pool, y_pool)
    elif models_parametrs.query_strategy_type == 'random':
        process_random(X_train, y_train, models_parametrs, X_pool, y_pool)
    

def train_committee(X_train, y_train, config, num_iter=60):
    committee = []
    for params in config.model_params.regressors_params:
        X_train = X_train.to_numpy() if isinstance(X_train, pd.DataFrame) else X_train
        y_train = y_train.to_numpy() if isinstance(y_train, pd.Series) else y_train
        train_x = torch.tensor(X_train, dtype=torch.float32)
        train_y = torch.tensor(y_train, dtype=torch.float32)
        likelihood = GaussianLikelihood()
        kernel = make_kernel(params)
        model = GPCommitteeModel(train_x, train_y, likelihood, kernel)
        optimizer = torch.optim.Adam(model.parameters(), lr=params.lr)
        mll = ExactMarginalLogLikelihood(likelihood, model)
        model.train()
        likelihood.train()
        for it in range(num_iter):
            optimizer.zero_grad()
            output = model(train_x)
            loss = -mll(output, train_y)
            if torch.isnan(loss):
                print(f"NaN in model {i}, iter {it}, breaking")
                break
            loss.backward()
            optimizer.step()
        committee.append((model, likelihood))
    return committee


def process_naqbc(X_train, y_train, models_parametrs, X_pool, y_pool):
    metrics_logger = MetricsLogger(
        save_dir=Path("C:/Users/ivan/VKR/experiments"),  
        name=models_parametrs.experiment_name,
        wandb_project="Active_learning"  
    )

    X_train = X_train.values if isinstance(X_train, pd.DataFrame) else X_train
    y_train = y_train.values if isinstance(y_train, pd.Series) else y_train
    X_pool = X_pool.values if isinstance(X_pool, pd.DataFrame) else X_pool
    for it in range(models_parametrs.n_start_points, models_parametrs.estimation_step):
        committee = train_committee(X_train, y_train, models_parametrs)
        best_grad, best_ind = None, None
        for rnd_indx in range(X_train.shape[0]):
            idx, grad = NA_query_strategy(committee, X_train[rnd_indx], X_pool)
            if best_grad is None:
                best_grad = grad
                best_ind = idx
            how_1 = comparison = [abs(g1) > abs(g2) for g1, g2 in zip(grad, best_grad)]
            if len(how_1) > len(best_grad):
                best_grad = grad
                best_ind = idx
        idx = best_ind
        X_train = np.vstack([X_train, X_pool[idx]])
        y_train = np.append(y_train, y_pool[idx])
        X_pool = np.delete(X_pool, idx, axis=0)
        y_pool = np.delete(y_pool, idx, axis=0)
        print(f"Step {it + 1}: added point {idx}, pool left: {len(X_pool)}")
        preds_list = []

        for model, likelihood in committee:
            model.eval()
            likelihood.eval()
            X_hold = torch.tensor(X_train, dtype=torch.float32)
            with torch.no_grad(), gpytorch.settings.fast_pred_var():
                preds = model(X_hold)
                y_pred = preds.mean.cpu().numpy().flatten()
                y_true = y_train
            preds_list.append(y_pred)
        y_pred = np.mean(preds_list, axis=0)
        mae, r2 = metrics_logger.log_metrics(
            y_true=y_true,
            y_pred=y_pred,
            name_csv=models_parametrs.experiment_name,
            iteration=it,
        )
        print(f"Final MAE: {mae:.4f}, R2: {r2:.4f}")
    metrics_logger.finalize()


def process_qbc(X_train, y_train, models_parametrs, X_pool, y_pool):
    metrics_logger = MetricsLogger(
        save_dir=Path("C:/Users/ivan/VKR/experiments"), 
        name=models_parametrs.experiment_name,
        wandb_project="Active_learning"  
    )

    X_train = X_train.values if isinstance(X_train, pd.DataFrame) else X_train
    y_train = y_train.values if isinstance(y_train, pd.Series) else y_train
    X_pool = X_pool.values if isinstance(X_pool, pd.DataFrame) else X_pool
    for it in range(models_parametrs.n_start_points, models_parametrs.estimation_step):
        committee = train_committee(X_train, y_train, models_parametrs)
    
    
        idx, x_dop = pool_based_qbc(committee, X_pool)
        X_train = np.vstack([X_train, X_pool[idx]])
        y_train = np.append(y_train, y_pool[idx])
        X_pool = np.delete(X_pool, idx, axis=0)
        y_pool = np.delete(y_pool, idx, axis=0)
        print(f"Step {it + 1}: added point {idx}, pool left: {len(X_pool)}")
        preds_list = []

        for model, likelihood in committee:
            model.eval()
            likelihood.eval()
            X_hold = torch.tensor(X_train, dtype=torch.float32)
            with torch.no_grad(), gpytorch.settings.fast_pred_var():
                preds = model(X_hold)
                y_pred = preds.mean.cpu().numpy().flatten()
                y_true = y_train
            preds_list.append(y_pred)
        y_pred = np.mean(preds_list, axis=0)
        mae, r2 = metrics_logger.log_metrics(
            y_true=y_true,
            y_pred=y_pred,
            name_csv=models_parametrs.experiment_name,
            iteration=it,
            
        )
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        print(f"Final MAE: {mae:.4f}, R2: {r2:.4f}")

    metrics_logger.finalize()

def process_random(X_train, y_train, models_parametrs, X_pool, y_pool):
    metrics_logger = MetricsLogger(
        save_dir=Path("C:/Users/ivan/VKR/experiments"),  # Папка для сохранения CSV
        name=models_parametrs.experiment_name,
        wandb_project="Active_learning"  # Имя проекта в wandb (None чтобы отключить)
    )

    X_train = X_train.values if isinstance(X_train, pd.DataFrame) else X_train
    y_train = y_train.values if isinstance(y_train, pd.Series) else y_train
    X_pool = X_pool.values if isinstance(X_pool, pd.DataFrame) else X_pool
    for it in range(models_parametrs.n_start_points, models_parametrs.estimation_step):
        committee = train_committee(X_train, y_train, models_parametrs)
    
    
        idx = np.random.choice(range(X_pool.shape[0]), size=1, replace=False)
        X_train = np.vstack([X_train, X_pool[idx]])
        y_train = np.append(y_train, y_pool[idx])
        X_pool = np.delete(X_pool, idx, axis=0)
        y_pool = np.delete(y_pool, idx, axis=0)
        print(f"Step {it + 1}: added point {idx}, pool left: {len(X_pool)}")
        preds_list = []

        for model, likelihood in committee:
            model.eval()
            likelihood.eval()
            X_hold = torch.tensor(X_train, dtype=torch.float32)
            with torch.no_grad(), gpytorch.settings.fast_pred_var():
                preds = model(X_hold)
                y_pred = preds.mean.cpu().numpy().flatten()
                y_true = y_train
            preds_list.append(y_pred)
        y_pred = np.mean(preds_list, axis=0)
        mae, r2 = metrics_logger.log_metrics(
            y_true=y_true,
            y_pred=y_pred,
            name_csv=models_parametrs.experiment_name,
            iteration=it,
            
        )
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        print(f"Final MAE: {mae:.4f}, R2: {r2:.4f}")

    metrics_logger.finalize()

def make_kernel(params) -> ScaleKernel:
    if params.kernel_type == 'RBF':
        kernel = ScaleKernel(RBFKernel(ard_num_dims=params.input_dim))
    elif params.kernel_type == 'Matern':
        kernel = ScaleKernel(MaternKernel(nu=params.nu, ard_num_dims=params.input_dim))
    return kernel

if __name__ == "__main__":
    active_lear()
