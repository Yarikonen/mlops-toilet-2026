# 📋 План реализации ML-пайплайна для Лабораторной работы 2

## 🎯 Требования лабораторной работы

### ✅ Обязательные требования:

1. **Использовать данные из Лабораторной 1** (туалеты, data.gov.au API)
2. **Реализовать базовое решение задачи МО**:
   - Замерить качество модели
   - Построить графики
3. **Сформулировать гипотезы по улучшению** и провести эксперименты
4. **Проанализировать результаты**
5. **Все эксперименты через трекер (MLflow)**:
   - Версия данных / идентификатор
   - Гиперпараметры
   - Кривые обучения
6. **Менеджер зависимостей** (Poetry/UV/аналоги)
7. **Код в виде модулей** (НЕ блокноты!)

---

## 🏗️ Архитектура решения

```
┌─────────────────────────────────────────────────────────────┐
│  DAG 1: toilet_data_pipeline (еженедельно)                  │
│  ─────────────────────────────────────────────────────────  │
│  ingest_api → merge_postgres → spark_aggregations          │
└─────────────────────────────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────┐
│  DAG 2: toilet_ml_pipeline (ежемесячно / manual)            │
│  ─────────────────────────────────────────────────────────  │
│  extract_features → train_model → log_mlflow → predict     │
└─────────────────────────────────────────────────────────────┘
```

---

## 📁 Структура проекта

```
mlops-toilet-2026/
├── pyproject.toml                    # NEW: Poetry/UV зависимости
├── docker-compose.yml
├── airflow/
│   ├── init-db.sql                   # + ml_predictions, ml_runs
│   ├── requirements.txt              # + mlflow, xgboost
│   └── dags/
│       ├── toilet_data_pipeline.py   # существующий + доработать
│       └── toilet_ml_pipeline.py     # NEW
├── spark_jobs/
│   ├── job.py                        # существующий
│   └── ml_features.py                # NEW: feature engineering
├── ml_models/                        # NEW модуль
│   ├── __init__.py
│   ├── config.py                     # конфиги MLflow, модели
│   ├── features.py                   # feature engineering
│   ├── train.py                      # обучение + MLflow
│   ├── predict.py                    # инференс
│   └── visualize.py                  # графики для отчёта
└── experiments/                      # NEW: отчёты по экспериментам
    ├── exp_01_baseline.md
    ├── exp_02_hyperparam.md
    └── exp_03_features.md
```

---

## 🔧 Шаг 1: Менеджер зависимостей (Poetry)

### Создать `pyproject.toml`:

```toml
[tool.poetry]
name = "toilet-ml-pipeline"
version = "0.1.0"
description = "ML pipeline for toilet location prediction"
authors = ["Your Name <your.email@example.com>"]
packages = [
    { include = "ml_models" },
]

[tool.poetry.dependencies]
python = ">=3.10,<3.13"
mlflow = "^2.0.0"
xgboost = "^2.0.0"
scikit-learn = "^1.3.0"
pandas = "^2.0.0"
numpy = "^1.24.0"
psycopg2-binary = "^2.9.9"
matplotlib = "^3.7.0"
seaborn = "^0.12.0"
jupyter = "^1.0.0"

[tool.poetry.group.dev.dependencies]
pytest = "^7.0.0"
black = "^23.0.0"
flake8 = "^6.0.0"

[build-system]
requires = ["poetry-core"]
build-backend = "poetry.core.masonry.api"
```

### Команды Poetry:
```bash
# Инициализация
poetry init  # или создать pyproject.toml вручную

# Установка зависимостей
poetry install

# Запуск скрипта в окружении Poetry
poetry run python ml_models/train.py

# Активация окружения
poetry shell
```

---

## 🔧 Шаг 2: Обновление БД

### Добавить в `airflow/init-db.sql`:

```sql
-- Таблица для предсказаний модели
CREATE TABLE IF NOT EXISTS ml_predictions (
    lat_bin BIGINT NOT NULL,
    lon_bin BIGINT NOT NULL,
    predicted_count INTEGER,
    actual_count INTEGER,
    gap INTEGER,
    model_version VARCHAR(50),
    run_id VARCHAR(100),
    predicted_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (lat_bin, lon_bin, run_id)
);

-- Таблица для метрик ML запусков
CREATE TABLE IF NOT EXISTS ml_runs (
    run_id VARCHAR(100) PRIMARY KEY,
    experiment_name VARCHAR(200),
    rmse DOUBLE PRECISION,
    mae DOUBLE PRECISION,
    r2 DOUBLE PRECISION,
    model_version VARCHAR(50),
    n_estimators INTEGER,
    max_depth INTEGER,
    learning_rate DOUBLE PRECISION,
    trained_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    status VARCHAR(50)
);

-- Индекс для быстрого поиска последних предсказаний
CREATE INDEX IF NOT EXISTS idx_ml_predictions_run_id 
ON ml_predictions(run_id);
```

---

## 🔧 Шаг 3: ML Features (Spark)

### Файл: `spark_jobs/ml_features.py`

```python
"""
Feature engineering для ML-модели предсказания туалетов.
Создаёт признаки на основе агрегированных данных.
"""
from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    col, sqrt, pow, when, count, avg, lit, concat_ws
)
import os

spark = SparkSession.builder \
    .appName("Toilet_ML_Features") \
    .config("spark.hadoop.fs.permissions.umask-mode", "000") \
    .master("spark://spark-master:7077") \
    .getOrCreate()

DB_HOST = os.getenv("DB_HOST", "postgres")
DB_PORT = os.getenv("DB_PORT", "5432")
DB_NAME = os.getenv("DB_NAME", "airflow")
DB_USER = os.getenv("DB_USER", "airflow")
DB_PASSWORD = os.getenv("DB_PASSWORD", "airflow")

JDBC_URL = f"jdbc:postgresql://{DB_HOST}:{DB_PORT}/{DB_NAME}"
JDBC_PROPERTIES = {
    "user": DB_USER,
    "password": DB_PASSWORD,
    "driver": "org.postgresql.Driver"
}

# Загрузка данных
print("Loading data from Postgres...")
raw_df = spark.read.jdbc(JDBC_URL, "raw_toilets", properties=JDBC_PROPERTIES)
grid_df = spark.read.jdbc(JDBC_URL, "toilets_grid_stats", properties=JDBC_PROPERTIES)
suburb_df = spark.read.jdbc(JDBC_URL, "toilets_suburb_stats", properties=JDBC_PROPERTIES)

print(f"Raw records: {raw_df.count()}")
print(f"Grid cells: {grid_df.count()}")
print(f"Suburbs: {suburb_df.count()}")

# Вычисление центра города
center = raw_df.agg(
    avg("Latitude").alias("center_lat"), 
    avg("Longitude").alias("center_lon")
).first()
CENTER_LAT = center["center_lat"] if center["center_lat"] else -31.95
CENTER_LON = center["center_lon"] if center["center_lon"] else 115.85

print(f"City center: ({CENTER_LAT}, {CENTER_LON})")

# Feature Engineering
print("Computing features...")

features_df = raw_df \
    .withColumn("distance_to_center",
                sqrt(
                    pow(col("Latitude") - CENTER_LAT, 2) + 
                    pow(col("Longitude") - CENTER_LON, 2)
                ) * 111000) \
    .withColumn("is_accessible",
                when(col("Accessible").isin(["Yes", "yes", "TRUE", "true"]), 1).otherwise(0)) \
    .withColumn("gender_encoded",
                when(col("Gender").contains("Unisex"), 0)
                .when(col("Gender").contains("Male"), 1)
                .when(col("Gender").contains("Female"), 2)
                .otherwise(3)) \
    .withColumn("lat_bin", (col("Latitude") * 100).cast("bigint")) \
    .withColumn("lon_bin", (col("Longitude") * 100).cast("bigint")) \
    .withColumn("category_encoded",
                when(col("Category").contains("Park"), 1)
                .when(col("Category").contains("Sport"), 2)
                .when(col("Category").contains("Shopping"), 3)
                .otherwise(0))

# Агрегация по grid для ML dataset
ml_dataset = features_df.groupBy("lat_bin", "lon_bin").agg(
    count("*").alias("toilet_count"),  # TARGET
    avg("distance_to_center").alias("avg_distance_to_center"),
    min("distance_to_center").alias("min_distance_to_center"),
    avg("is_accessible").alias("accessibility_rate"),
    avg("gender_encoded").alias("avg_gender_encoded"),
    avg("category_encoded").alias("avg_category_encoded"),
    count(when(col("Category").contains("Park"), 1)).alias("park_count"),
    count(when(col("Category").contains("Sport"), 1)).alias("sport_count"),
    count(when(col("Category").contains("Shopping"), 1)).alias("shopping_count")
)

# Join с suburb stats
ml_dataset = ml_dataset.join(
    suburb_df.select(
        "Suburb", 
        "accessibility_rate".alias("suburb_accessibility"),
        "toilet_count".alias("suburb_toilet_count")
    ).distinct(),
    on="Suburb",
    how="left"
)

# Сохранение
print("Saving ML features...")
ml_dataset.write.mode("overwrite").parquet("/opt/data/processed/ml_features")
ml_dataset.write.jdbc(
    url=JDBC_URL,
    table="ml_features",
    mode="overwrite",
    properties=JDBC_PROPERTIES
)

print(f"ML dataset created: {ml_dataset.count()} records")
spark.stop()
```

---

## 🔧 Шаг 4: ML Module

### Файл: `ml_models/config.py`

```python
"""
Конфигурация ML-пайплайна.
"""
from dataclasses import dataclass
from typing import List, Dict

@dataclass
class MLflowConfig:
    TRACKING_URI: str = "http://localhost:5000"
    EXPERIMENT_NAME: str = "toilet_location_prediction"
    ARTIFACT_PATH: str = "model"

@dataclass
class ModelConfig:
    """Гиперпараметры XGBoost по умолчанию"""
    n_estimators: int = 100
    max_depth: int = 6
    learning_rate: float = 0.1
    subsample: float = 0.8
    colsample_bytree: float = 0.8
    random_state: int = 42

@dataclass
class DBConfig:
    HOST: str = "postgres"
    PORT: str = "5432"
    NAME: str = "airflow"
    USER: str = "airflow"
    PASSWORD: str = "airflow"

FEATURE_COLUMNS: List[str] = [
    "avg_distance_to_center",
    "min_distance_to_center",
    "accessibility_rate",
    "avg_gender_encoded",
    "avg_category_encoded",
    "park_count",
    "sport_count",
    "shopping_count",
]

TARGET_COLUMN: str = "toilet_count"
```

---

### Файл: `ml_models/features.py`

```python
"""
Feature engineering и загрузка данных.
"""
import pandas as pd
import numpy as np
from typing import Tuple, List
import psycopg2

from ml_models.config import DBConfig, FEATURE_COLUMNS, TARGET_COLUMN


def load_features_from_postgres() -> pd.DataFrame:
    """Загрузка признаков из Postgres"""
    conn = psycopg2.connect(
        host=DBConfig.HOST,
        port=DBConfig.PORT,
        database=DBConfig.NAME,
        user=DBConfig.USER,
        password=DBConfig.PASSWORD
    )
    
    feature_cols_str = ", ".join(["lat_bin", "lon_bin"] + FEATURE_COLUMNS + [TARGET_COLUMN])
    query = f"SELECT {feature_cols_str} FROM ml_features"
    
    df = pd.read_sql_query(query, conn)
    conn.close()
    
    print(f"Loaded {len(df)} records from Postgres")
    return df


def load_features_from_parquet(path: str = "/opt/data/processed/ml_features") -> pd.DataFrame:
    """Загрузка признаков из Parquet"""
    df = pd.read_parquet(path)
    print(f"Loaded {len(df)} records from Parquet")
    return df


def prepare_features(
    df: pd.DataFrame, 
    feature_cols: List[str] = FEATURE_COLUMNS
) -> Tuple[pd.DataFrame, pd.Series, List[str]]:
    """
    Подготовка признаков и таргета.
    
    Returns:
        X: Features DataFrame
        y: Target Series
        feature_cols: List of feature column names
    """
    # Заполнение пропусков
    df[feature_cols] = df[feature_cols].fillna(0)
    
    # Замена бесконечных значений
    df[feature_cols] = df[feature_cols].replace([np.inf, -np.inf], 0)
    
    X = df[feature_cols]
    y = df[TARGET_COLUMN]
    
    return X, y, feature_cols


def get_feature_statistics(df: pd.DataFrame) -> pd.DataFrame:
    """Статистика по признакам"""
    stats = df[FEATURE_COLUMNS].describe()
    return stats
```

---

### Файл: `ml_models/train.py`

```python
"""
Обучение ML-модели с логированием в MLflow.
"""
import mlflow
import mlflow.xgboost
import xgboost as xgb
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (
    mean_squared_error, 
    mean_absolute_error, 
    r2_score,
    median_absolute_error
)
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from datetime import datetime
import argparse

from ml_models.config import MLflowConfig, ModelConfig, DBConfig
from ml_models.features import (
    load_features_from_postgres,
    prepare_features,
    get_feature_statistics
)


def plot_feature_importance(model, feature_names: list, save_path: str = None):
    """График важности признаков"""
    fig, ax = plt.subplots(figsize=(10, 8))
    xgb.plot_importance(model, ax=ax, importance_type='gain')
    plt.title("Feature Importance (Gain)")
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
    return fig


def plot_actual_vs_predicted(y_true, y_pred, r2: float, save_path: str = None):
    """График Actual vs Predicted"""
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(y_true, y_pred, alpha=0.5, edgecolors='k', linewidth=0.5)
    
    # Линия идеального предсказания
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', label='Perfect prediction')
    
    ax.set_xlabel("Actual toilet count")
    ax.set_ylabel("Predicted toilet count")
    ax.set_title(f"Actual vs Predicted (R² = {r2:.3f})")
    ax.legend()
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
    return fig


def plot_learning_curve(model, X_train, y_train, save_path: str = None):
    """Кривая обучения"""
    train_sizes, train_scores, test_scores = [], []
    
    # Упрощённая версия: используем cross-validation
    from sklearn.model_selection import learning_curve
    train_sizes, train_scores, test_scores = learning_curve(
        model, X_train, y_train, 
        cv=5, 
        train_sizes=np.linspace(0.1, 1.0, 10),
        scoring='r2'
    )
    
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(train_sizes, train_scores.mean(axis=1), 'o-', label='Training score')
    ax.plot(train_sizes, test_scores.mean(axis=1), 'o-', label='Cross-validation score')
    ax.set_xlabel("Training examples")
    ax.set_ylabel("R² score")
    ax.set_title("Learning Curve")
    ax.legend()
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
    return fig


def train_model(
    experiment_name: str = None,
    hyperparams: dict = None,
    use_parquet: bool = False
):
    """
    Обучение XGBoost модели с логированием в MLflow.
    
    Args:
        experiment_name: Название эксперимента в MLflow
        hyperparams: Гиперпараметры модели (если None, используются default)
        use_parquet: Загружать данные из Parquet вместо Postgres
    """
    # Инициализация MLflow
    mlflow.set_tracking_uri(MLflowConfig.TRACKING_URI)
    mlflow.set_experiment(experiment_name or MLflowConfig.EXPERIMENT_NAME)
    
    # Загрузка данных
    if use_parquet:
        df = load_features_from_parquet()
    else:
        df = load_features_from_postgres()
    
    # Статистика
    print("\nFeature statistics:")
    print(get_feature_statistics(df))
    
    # Подготовка
    X, y, feature_cols = prepare_features(df)
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=ModelConfig.random_state
    )
    
    print(f"\nTrain size: {len(X_train)}, Test size: {len(X_test)}")
    
    # Параметры модели
    params = ModelConfig.__dict__ if hyperparams is None else hyperparams
    params = {k: v for k, v in params.items() if not k.startswith('_')}
    
    # Обучение с MLflow
    with mlflow.start_run() as run:
        print(f"\nStarting MLflow run: {run.info.run_id}")
        
        model = xgb.XGBRegressor(**params)
        model.fit(X_train, y_train)
        
        # Предсказания
        y_pred_train = model.predict(X_train)
        y_pred = model.predict(X_test)
        
        # Метрики
        metrics = {
            "rmse": np.sqrt(mean_squared_error(y_test, y_pred)),
            "mae": mean_absolute_error(y_test, y_pred),
            "r2": r2_score(y_test, y_pred),
            "median_ae": median_absolute_error(y_test, y_pred),
            "train_rmse": np.sqrt(mean_squared_error(y_train, y_pred_train)),
            "train_r2": r2_score(y_train, y_pred_train)
        }
        
        # Логирование параметров и метрик
        mlflow.log_params(params)
        mlflow.log_metrics(metrics)
        
        # Логирование модели
        mlflow.xgboost.log_model(model, MLflowConfig.ARTIFACT_PATH)
        
        # Графики
        fig1 = plot_feature_importance(model, feature_cols)
        mlflow.log_figure(fig1, "feature_importance.png")
        plt.close(fig1)
        
        fig2 = plot_actual_vs_predicted(y_test, y_pred, metrics["r2"])
        mlflow.log_figure(fig2, "actual_vs_predicted.png")
        plt.close(fig2)
        
        fig3 = plot_learning_curve(model, X_train, y_train)
        mlflow.log_figure(fig3, "learning_curve.png")
        plt.close(fig3)
        
        # Логирование в Postgres
        log_metrics_to_postgres(run.info.run_id, metrics, params)
        
        print(f"\n✅ Experiment completed!")
        print(f"   RMSE: {metrics['rmse']:.3f}")
        print(f"   MAE: {metrics['mae']:.3f}")
        print(f"   R²: {metrics['r2']:.3f}")
        print(f"   Run ID: {run.info.run_id}")
        
        return model, run.info.run_id, metrics


def log_metrics_to_postgres(run_id: str, metrics: dict, params: dict):
    """Логирование метрик в Postgres"""
    conn = psycopg2.connect(
        host=DBConfig.HOST,
        port=DBConfig.PORT,
        database=DBConfig.NAME,
        user=DBConfig.USER,
        password=DBConfig.PASSWORD
    )
    cursor = conn.cursor()
    
    cursor.execute("""
        INSERT INTO ml_runs (run_id, experiment_name, rmse, mae, r2,
                            model_version, n_estimators, max_depth, learning_rate, status)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        ON CONFLICT (run_id) DO UPDATE SET
            rmse = EXCLUDED.rmse,
            mae = EXCLUDED.mae,
            r2 = EXCLUDED.r2,
            status = EXCLUDED.status
    """, (
        run_id,
        MLflowConfig.EXPERIMENT_NAME,
        metrics.get("rmse", 0),
        metrics.get("mae", 0),
        metrics.get("r2", 0),
        datetime.now().strftime("%Y%m%d_%H%M%S"),
        params.get("n_estimators", 100),
        params.get("max_depth", 6),
        params.get("learning_rate", 0.1),
        "success"
    ))
    
    conn.commit()
    cursor.close()
    conn.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train toilet prediction model")
    parser.add_argument("--experiment", type=str, default=None, help="Experiment name")
    parser.add_argument("--parquet", action="store_true", help="Use Parquet files")
    args = parser.parse_args()
    
    train_model(experiment_name=args.experiment, use_parquet=args.parquet)
```

---

### Файл: `ml_models/predict.py`

```python
"""
Инференс и анализ gaps (где не хватает туалетов).
"""
import mlflow
import pandas as pd
import psycopg2
from datetime import datetime
import argparse

from ml_models.config import MLflowConfig, DBConfig
from ml_models.features import load_features_from_postgres, prepare_features


def predict_gaps(model_uri: str = None):
    """
    Предсказание и анализ gaps.
    
    Gap = predicted - actual
    Gap > 0: туалетов должно быть больше, чем есть (нужно строить)
    Gap < 0: туалетов больше, чем нужно (избыток)
    """
    mlflow.set_tracking_uri(MLflowConfig.TRACKING_URI)
    
    # Загрузка последней модели
    if model_uri is None:
        runs = mlflow.search_runs(
            order_by=["start_time DESC"], 
            max_results=1
        )
        if len(runs) == 0:
            raise Exception("No trained models found in MLflow")
        run_id = runs.iloc[0]["run_id"]
        model_uri = f"runs:/{run_id}/model"
        print(f"Using model from run: {run_id}")
    
    model = mlflow.xgboost.load_model(model_uri)
    
    # Загрузка данных
    df = load_features_from_postgres()
    X, y, feature_cols = prepare_features(df)
    
    # Предсказания
    predictions = model.predict(X)
    
    # Расчет gaps
    df["predicted_count"] = predictions.astype(int)
    df["actual_count"] = y.values.astype(int)
    df["gap"] = df["predicted_count"] - df["actual_count"]
    
    # Сохранение в Postgres
    conn = psycopg2.connect(
        host=DBConfig.HOST,
        port=DBConfig.PORT,
        database=DBConfig.NAME,
        user=DBConfig.USER,
        password=DBConfig.PASSWORD
    )
    cursor = conn.cursor()
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    for _, row in df.iterrows():
        cursor.execute("""
            INSERT INTO ml_predictions (lat_bin, lon_bin, predicted_count,
                                        actual_count, gap, model_version, run_id)
            VALUES (%s, %s, %s, %s, %s, %s, %s)
        """, (
            int(row["lat_bin"]), 
            int(row["lon_bin"]),
            int(row["predicted_count"]), 
            int(row["actual_count"]),
            int(row["gap"]), 
            timestamp, 
            run_id
        ))
    
    conn.commit()
    cursor.close()
    conn.close()
    
    # Анализ результатов
    print("\n" + "="*50)
    print("GAP ANALYSIS")
    print("="*50)
    print(f"Total grid cells: {len(df)}")
    print(f"Cells with gap > 0 (need more): {(df['gap'] > 0).sum()}")
    print(f"Cells with gap < 0 (oversupplied): {(df['gap'] < 0).sum()}")
    
    print("\n🔴 TOP-10 ZONES NEED MORE TOILETS:")
    top_gaps = df.nlargest(10, "gap")[
        ["lat_bin", "lon_bin", "predicted_count", "actual_count", "gap"]
    ]
    print(top_gaps.to_string(index=False))
    
    print("\n🟢 TOP-10 ZONES WITH EXCESS TOILETS:")
    bottom_gaps = df.nsmallest(10, "gap")[
        ["lat_bin", "lon_bin", "predicted_count", "actual_count", "gap"]
    ]
    print(bottom_gaps.to_string(index=False))
    
    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict toilet gaps")
    parser.add_argument("--model-uri", type=str, default=None, help="MLflow model URI")
    args = parser.parse_args()
    
    predict_gaps(model_uri=args.model_uri)
```

---

### Файл: `ml_models/visualize.py`

```python
"""
Визуализация результатов для отчёта.
"""
import psycopg2
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from ml_models.config import DBConfig


def plot_gap_distribution():
    """Распределение gaps"""
    conn = psycopg2.connect(
        host=DBConfig.HOST,
        port=DBConfig.PORT,
        database=DBConfig.NAME,
        user=DBConfig.USER,
        password=DBConfig.PASSWORD
    )
    
    query = """
        SELECT gap, predicted_count, actual_count
        FROM ml_predictions
        WHERE run_id = (SELECT run_id FROM ml_predictions ORDER BY predicted_at DESC LIMIT 1)
    """
    
    df = pd.read_sql_query(query, conn)
    conn.close()
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Histogram
    axes[0].hist(df['gap'], bins=50, edgecolor='black', alpha=0.7)
    axes[0].axvline(x=0, color='r', linestyle='--', label='Zero gap')
    axes[0].set_xlabel('Gap (predicted - actual)')
    axes[0].set_ylabel('Count')
    axes[0].set_title('Distribution of Gaps')
    axes[0].legend()
    
    # Scatter
    axes[1].scatter(df['actual_count'], df['predicted_count'], alpha=0.5)
    min_val = min(df['actual_count'].min(), df['predicted_count'].min())
    max_val = max(df['actual_count'].max(), df['predicted_count'].max())
    axes[1].plot([min_val, max_val], [min_val, max_val], 'r--')
    axes[1].set_xlabel('Actual count')
    axes[1].set_ylabel('Predicted count')
    axes[1].set_title('Actual vs Predicted')
    
    plt.tight_layout()
    plt.savefig("/opt/data/processed/gap_analysis.png", dpi=150)
    print("Saved: /opt/data/processed/gap_analysis.png")


def plot_mlflow_comparison():
    """Сравнение экспериментов из MLflow"""
    import mlflow
    
    mlflow.set_tracking_uri(MLflowConfig.TRACKING_URI)
    
    runs = mlflow.search_runs(order_by=["start_time ASC"])
    
    if len(runs) == 0:
        print("No experiments found")
        return
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    # Метрики по запускам
    axes[0].plot(range(len(runs)), runs['metrics.r2'], 'o-', label='R²')
    axes[0].set_xlabel('Run #')
    axes[0].set_ylabel('Score')
    axes[0].set_title('R² by Run')
    axes[0].grid(True, alpha=0.3)
    
    axes[1].plot(range(len(runs)), runs['metrics.rmse'], 'o-', color='orange', label='RMSE')
    axes[1].set_xlabel('Run #')
    axes[1].set_ylabel('RMSE')
    axes[1].set_title('RMSE by Run')
    axes[1].grid(True, alpha=0.3)
    
    # Параметры
    if 'params.max_depth' in runs.columns:
        axes[2].scatter(runs['params.max_depth'], runs['metrics.r2'], alpha=0.7)
        axes[2].set_xlabel('Max Depth')
        axes[2].set_ylabel('R²')
        axes[2].set_title('Max Depth vs R²')
    
    plt.tight_layout()
    plt.savefig("/opt/data/processed/mlflow_comparison.png", dpi=150)
    print("Saved: /opt/data/processed/mlflow_comparison.png")


if __name__ == "__main__":
    plot_gap_distribution()
    plot_mlflow_comparison()
```

---

## 🔧 Шаг 5: Airflow DAGs

### Файл: `airflow/dags/toilet_ml_pipeline.py`

```python
"""
ML Pipeline для предсказания размещения туалетов.
"""
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.bash import BashOperator
from airflow.providers.apache.spark.operators.spark_submit import SparkSubmitOperator
from datetime import datetime
import sys

sys.path.append("/opt/ml_models")

with DAG(
    dag_id="toilet_ml_pipeline",
    start_date=datetime(2024, 1, 1),
    schedule="@monthly",
    catchup=False,
    default_args={"retries": 1, "retry_delay": 300},
    doc_md=__doc__
) as dag:
    
    # Шаг 1: Feature engineering через Spark
    extract_features = SparkSubmitOperator(
        task_id="extract_features",
        application="/opt/spark_jobs/ml_features.py",
        conn_id="spark_default",
        jars="/opt/spark/jars/postgresql-42.6.0.jar",
        executor_memory="2g",
        driver_memory="1g",
        verbose=True
    )
    
    # Шаг 2: Обучение модели
    train_model = PythonOperator(
        task_id="train_model",
        python_callable=lambda: exec(open("/opt/ml_models/train.py").read()),
        op_kwargs={"experiment_name": "toilet_location_airflow"}
    )
    
    # Шаг 3: Предсказание и gap analysis
    predict_gaps = PythonOperator(
        task_id="predict_gaps",
        python_callable=lambda: exec(open("/opt/ml_models/predict.py").read())
    )
    
    # Шаг 4: Визуализация
    visualize = PythonOperator(
        task_id="visualize",
        python_callable=lambda: exec(open("/opt/ml_models/visualize.py").read())
    )
    
    # Шаг 5: Уведомление
    notify = BashOperator(
        task_id="notify_complete",
        bash_command="echo 'ML Pipeline completed successfully!'"
    )
    
    extract_features >> train_model >> predict_gaps >> visualize >> notify
```

---

## 📊 Шаг 6: Эксперименты

### План экспериментов для отчёта:

#### Эксперимент 1: Baseline модель
- **Цель**: Получить базовые метрики
- **Модель**: XGBoost с параметрами по умолчанию
- **Гиперпараметры**: n_estimators=100, max_depth=6, learning_rate=0.1
- **Метрики**: RMSE, MAE, R²

#### Эксперимент 2: Подбор гиперпараметров
- **Цель**: Улучшить метрики
- **Вариации**:
  - max_depth: [4, 6, 8, 10]
  - learning_rate: [0.01, 0.05, 0.1, 0.2]
  - n_estimators: [50, 100, 200]
- **Метрики**: Сравнение R², RMSE

#### Эксперимент 3: Feature ablation
- **Цель**: Оценить важность признаков
- **Вариации**: Удалять признаки по одному
- **Метрики**: Изменение R²

---

## 📝 Шаг 7: Отчёт по экспериментам

### Шаблон: `experiments/exp_01_baseline.md`

```markdown
# Эксперимент 1: Baseline модель

## Дата проведения
2026-05-03

## Цель
Получить базовые метрики модели предсказания количества туалетов

## Данные
- Источник: ml_features (Postgres)
- Количество записей: XXXX
- Признаки: avg_distance_to_center, accessibility_rate, ...

## Модель
XGBoost Regressor

## Гиперпараметры
- n_estimators: 100
- max_depth: 6
- learning_rate: 0.1
- subsample: 0.8
- colsample_bytree: 0.8

## Метрики
| Метрика | Значение |
|---------|----------|
| RMSE (test) | X.XX |
| MAE (test) | X.XX |
| R² (test) | 0.XX |
| RMSE (train) | X.XX |
| R² (train) | 0.XX |

## Графики
- feature_importance.png
- actual_vs_predicted.png
- learning_curve.png

## Выводы
Модель показывает [хорошие/удовлетворительные] результаты.
R² = 0.XX означает, что модель объясняет XX% дисперсии.
```

---

## 🚀 Запуск пайплайна

```bash
# 1. Сборка и запуск инфраструктуры
docker-compose up -d --build

# 2. Инициализация Poetry (в проекте)
poetry install

# 3. Запуск MLflow UI
poetry run mlflow ui --host 0.0.0.0 --port 5000

# 4. Активация DAGs в Airflow UI (http://localhost:8080)
#    - toilet_data_pipeline
#    - toilet_ml_pipeline

# 5. Запуск обучения локально (опционально)
poetry run python ml_models/train.py

# 6. Запуск предсказания
poetry run python ml_models/predict.py

# 7. Визуализация
poetry run python ml_models/visualize.py
```

---

## ✅ Чеклист соответствия требованиям

| Требование | Статус | Реализация |
|------------|--------|------------|
| Данные из Лабы 1 | ✅ | data.gov.au API, туалеты |
| Базовое решение МО | ✅ | XGBoost Regressor |
| Замер качества | ✅ | RMSE, MAE, R² |
| Графики | ✅ | actual_vs_predicted, feature_importance, learning_curve |
| Гипотезы и эксперименты | ✅ | 3 эксперимента (baseline, hyperparams, features) |
| Трекер экспериментов | ✅ | MLflow |
| Версия данных | ✅ | run_id + timestamp в ml_runs |
| Гиперпараметры | ✅ | Логирование в MLflow |
| Кривые обучения | ✅ | learning_curve.png |
| Менеджер зависимостей | ✅ | Poetry |
| Код модулями | ✅ | ml_models/ пакет |

---

## 🔗 Полезные ссылки

- [MLflow Documentation](https://mlflow.org/docs/latest/index.html)
- [XGBoost Documentation](https://xgboost.readthedocs.io/)
- [Poetry Documentation](https://python-poetry.org/docs/)
- [Airflow Spark Operator](https://airflow.apache.org/docs/apache-airflow-providers-apache-spark/stable/)
