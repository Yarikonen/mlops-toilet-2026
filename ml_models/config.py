"""ml_models/config.py

Конфигурация ML-пайплайна (строго по плану).
Настройки читаются из переменных окружения, чтобы одинаково работать локально и в Docker.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import List


@dataclass(frozen=True)
class MLflowConfig:
    TRACKING_URI: str = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
    EXPERIMENT_NAME: str = os.getenv("MLFLOW_EXPERIMENT_NAME", "toilet_location_prediction")
    ARTIFACT_PATH: str = os.getenv("MLFLOW_MODEL_ARTIFACT_PATH", "model")


@dataclass(frozen=True)
class ModelConfig:
    """Гиперпараметры XGBoost по умолчанию (baseline)."""

    n_estimators: int = int(os.getenv("XGB_N_ESTIMATORS", "100"))
    max_depth: int = int(os.getenv("XGB_MAX_DEPTH", "6"))
    learning_rate: float = float(os.getenv("XGB_LEARNING_RATE", "0.1"))
    subsample: float = float(os.getenv("XGB_SUBSAMPLE", "0.8"))
    colsample_bytree: float = float(os.getenv("XGB_COLSAMPLE_BYTREE", "0.8"))
    random_state: int = int(os.getenv("XGB_RANDOM_STATE", "42"))


@dataclass(frozen=True)
class DBConfig:
    HOST: str = os.getenv("DB_HOST", "postgres")
    PORT: str = os.getenv("DB_PORT", "5432")
    NAME: str = os.getenv("DB_NAME", "airflow")
    USER: str = os.getenv("DB_USER", "airflow")
    PASSWORD: str = os.getenv("DB_PASSWORD", "airflow")


PARQUET_FEATURES_PATH: str = os.getenv("ML_FEATURES_PARQUET_PATH", "/opt/data/processed/ml_features")


FEATURE_COLUMNS: List[str] = [
    "avg_distance_to_center",
    "min_distance_to_center",
    "accessibility_rate",
    "avg_gender_encoded",
    "avg_category_encoded",
    "park_count",
    "sport_count",
    "shopping_count",
    "suburb_accessibility",
    "suburb_toilet_count",
]

TARGET_COLUMN: str = "toilet_count"

# Used for lightweight identification in Postgres tables
MODEL_VERSION: str = os.getenv("MODEL_VERSION", "xgboost")
