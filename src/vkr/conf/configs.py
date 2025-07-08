from dataclasses import dataclass
from typing import Union


@dataclass
class DataConfig:
    raw_data_path: str
    normalized_data_path: str
    train_data_path: str
    val_data_path: str
    val_size: float
    params_ranges: dict[str, list[float, float]]


@dataclass
class BaselineTrainConfig:
    train_data_path: str
    val_data_path: str
    normalized_data_path: str
    experiment_name: str
    model_info_dir: str


@dataclass
class ModelConfig:
    type_model: str
    parameters: dict[str, Union[int, float]]
    model_info_dir: str


@dataclass
class NnTrainConfig:
    artifact_dir: str
    model_dir: str


@dataclass
class ALTrainConfig:
    register_model_name: str
    model_info_dir: str
    query_strategy_type: str
    model_type: str
    model_params: dict[str, Union[int, float]]
    n_start_points: int
    estimation_step: int
    experiment_name: str