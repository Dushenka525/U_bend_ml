import numpy as np
import torch
import gpytorch
import pandas as pd
from joblib import dump, load
from sklearn.metrics import mean_absolute_error, r2_score
from typing import Union
from gpytorch.models import ExactGP
from gpytorch.means import ConstantMean
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import SGDRegressor
from gpytorch.kernels import ScaleKernel, RBFKernel, MaternKernel, AdditiveKernel, LinearKernel
from gpytorch.likelihoods import GaussianLikelihood
from kan import KAN, create_dataset_from_data
from sklearn.base import BaseEstimator, RegressorMixin
from pathlib import Path




class SerializableKANWrapper(BaseEstimator, RegressorMixin):
    def __init__(self, width=[7,7,7,1], grid=7, k=3, seed=42, 
                 lr=1e-3, epochs=1500, opt="lbfgs", lamb=0.0):
        self.width = width
        self.grid = grid
        self.k = k
        self.seed = seed
        self.lr = lr
        self.epochs = epochs
        self.opt = opt
        self.lamb = lamb
        self._model = None
        self._optimizer = None
    
    def _convert_to_tensor(self, data):
        """Конвертирует pandas DataFrame/Series в torch.Tensor"""
        if isinstance(data, (pd.DataFrame, pd.Series)):
            data = data.values
        if isinstance(data, np.ndarray):
            data = torch.from_numpy(data)
        return data.float()
    
    def _init_model(self, input_dim):
        if self._model is None:
            torch.manual_seed(self.seed)
            self._model = KAN(width=[input_dim] + self.width[1:], grid=self.grid, k=self.k)
            
            if self.opt.lower() == "adam":
                self._optimizer = torch.optim.Adam(self._model.parameters(), lr=self.lr)
            elif self.opt.lower() == "lbfgs":
                self._optimizer = torch.optim.LBFGS(self._model.parameters(), 
                                                  lr=self.lr, 
                                                  max_iter=self.epochs,
                                                  history_size=100)
    
    def fit(self, X, y):
        X_tensor = self._convert_to_tensor(X)
        y_tensor = self._convert_to_tensor(y)
        self._init_model(X_tensor.shape[1])
        
        self._model.train()
        
        if self.opt.lower() == "lbfgs":
            def closure():
                self._optimizer.zero_grad()
                outputs = self._model(X_tensor)
                loss = torch.nn.functional.mse_loss(outputs, y_tensor.unsqueeze(1))
                if self.lamb > 0:
                    l2_reg = sum(p.norm()**2 for p in self._model.parameters())
                    loss += self.lamb * l2_reg
                loss.backward()
                return loss
            self._optimizer.step(closure)
        else:
            for epoch in range(self.epochs):
                self._optimizer.zero_grad()
                outputs = self._model(X_tensor)
                loss = torch.nn.functional.mse_loss(outputs, y_tensor.unsqueeze(1))
                if self.lamb > 0:
                    l2_reg = sum(p.norm()**2 for p in self._model.parameters())
                    loss += self.lamb * l2_reg
                loss.backward()
                self._optimizer.step()
        return self
    
    def predict(self, X):
        """Обязательный метод для scikit-learn совместимости"""
        if self._model is None:
            raise RuntimeError("Model not trained yet. Call fit() first.")
            
        X_tensor = self._convert_to_tensor(X)
        self._model.eval()
        with torch.no_grad():
            predictions = self._model(X_tensor)
            return predictions.numpy().flatten()
    
    def save(self, path):
        """Сохранение модели с проверкой атрибутов"""
        if not hasattr(self, '_model'):
            raise RuntimeError("Model not initialized")
            
        torch.save({
            'width': self.width,
            'grid': self.grid,
            'k': self.k,
            'state_dict': self._model.state_dict(),
            'opt': self.opt,
            'lamb': self.lamb
        }, path)
    
    @classmethod
    def load(cls, path):
        """Загрузка модели с проверкой атрибутов"""
        data = torch.load(path)
        wrapper = cls(
            width=data['width'],
            grid=data['grid'],
            k=data['k'],
            opt=data.get('opt', 'Adam'),
            lamb=data.get('lamb', 0.0)
        )
        wrapper._init_model(data['width'][0])
        wrapper._model.load_state_dict(data['state_dict'])
        return wrapper



class GPRegressor:
    def __init__(self, nu=1.5, lr=0.1, n_epochs=100):
        self.nu = nu
        self.lr = lr
        self.n_epochs = n_epochs
        self.model = None
        self.likelihood = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def _ensure_tensor(self, data):
        """Конвертирует данные в torch.Tensor и переносит на нужное устройство"""
        if isinstance(data, (pd.DataFrame, pd.Series)):
            data = data.values
        if isinstance(data, np.ndarray):
            data = torch.from_numpy(data)
        if isinstance(data, torch.Tensor):
            return data.float().to(self.device)
        raise ValueError(f"Unsupported data type: {type(data)}")

    def fit(self, train_X, train_y):
        
        train_X = self._ensure_tensor(train_X)
        train_y = self._ensure_tensor(train_y)
        
        if train_X.dim() == 1:
            train_X = train_X.unsqueeze(-1)
        if train_y.dim() > 1:
            train_y = train_y.squeeze()
        
        self.likelihood = GaussianLikelihood().to(self.device)
        self.model = GPRegressionModel(train_X, train_y, self.likelihood).to(self.device)
        
        self.model.train()
        self.likelihood.train()
        
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(self.likelihood, self.model)

        for epoch in range(self.n_epochs):
            optimizer.zero_grad()
            output = self.model(train_X)
            loss = -mll(output, train_y)
            loss.backward()
            optimizer.step()
        
        return self

    def predict(self, X, return_std=False):
        if self.model is None:
            raise RuntimeError("Model not trained. Call fit() first.")
            
        X = self._ensure_tensor(X)
        if X.dim() == 1:
            X = X.unsqueeze(-1)
        
        self.model.eval()
        self.likelihood.eval()
        
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            observed_pred = self.likelihood(self.model(X))
            mean = observed_pred.mean.cpu().numpy()
            
            if return_std:
                std = observed_pred.stddev.cpu().numpy()
                return mean, std
            return mean


class GPRegressionModel(ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super().__init__(train_x, train_y, likelihood)
        self.mean_module = ConstantMean()
        self.covar_module = (
            ScaleKernel(MaternKernel(nu=1.5, ard_num_dims=train_x.shape[1])) +
            ScaleKernel(RBFKernel(ard_num_dims=train_x.shape[1]))
        )

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
    
# --- Обёртка для одной модели ---
class GPCommitteeModel(ExactGP):
    def __init__(self, train_x, train_y, likelihood, kernel):
        super().__init__(train_x, train_y, likelihood)
        self.mean_module = ConstantMean()
        self.covar_module = kernel

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
    

def baseline_models(
    gpr_default_params: dict[str, Union[int, float]],
) -> dict[
    str,
    Union[
        SGDRegressor,
        GradientBoostingRegressor,
        RandomForestRegressor,
        GPRegressor,
        KAN,
    ],
]:
    base_gpr = GPRegressor(**gpr_default_params)
    return {
        "linreg": SGDRegressor(),
        "gboost": GradientBoostingRegressor(),
        "random_forest": RandomForestRegressor(),
        "gpr": base_gpr,
        "kan": SerializableKANWrapper(),
    }