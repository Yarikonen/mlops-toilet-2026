"""ml_models/train.py

Обучение baseline ML-модели (XGBoost Regressor) с логированием в MLflow.
Также пишет ключевые метрики в Postgres table `ml_runs`.

Запуск локально:
  poetry run python -m ml_models.train

В Docker/Airflow используется как python_callable.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
from typing import Any, Dict, Optional, Tuple

import matplotlib.pyplot as plt
import mlflow
import mlflow.xgboost
import numpy as np
import pandas as pd
import psycopg2
import xgboost as xgb
from pandas.util import hash_pandas_object
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import learning_curve, train_test_split

from ml_models.config import DBConfig, MLflowConfig, MODEL_VERSION, ModelConfig
from ml_models.features import (
    get_feature_statistics,
    load_features_from_parquet,
    load_features_from_postgres,
    prepare_features,
)


def _connect():
    return psycopg2.connect(
        host=DBConfig.HOST,
        port=DBConfig.PORT,
        database=DBConfig.NAME,
        user=DBConfig.USER,
        password=DBConfig.PASSWORD,
    )


def _data_fingerprint(df: pd.DataFrame) -> str:
    # Cheap-ish stable fingerprint for experiment tracking (requirement: version/identifier)
    cols = [c for c in df.columns if c not in {"created_at"}]
    hashed = hash_pandas_object(df[cols], index=False).values
    return hex(int(hashed.sum()) & 0xFFFFFFFFFFFF)


def plot_feature_importance(model: xgb.XGBRegressor, feature_names: list) -> plt.Figure:
    importances = getattr(model, "feature_importances_", None)
    fig, ax = plt.subplots(figsize=(10, 6))

    if importances is None:
        ax.set_title("Feature importance not available")
        return fig

    order = np.argsort(importances)[::-1]
    ax.bar([feature_names[i] for i in order], importances[order])
    ax.set_title("Feature Importance")
    ax.set_ylabel("importance")
    ax.tick_params(axis="x", rotation=45, labelsize=9)
    plt.tight_layout()
    return fig


def plot_actual_vs_predicted(y_true: np.ndarray, y_pred: np.ndarray, r2: float) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(y_true, y_pred, alpha=0.5, edgecolors="k", linewidth=0.3)

    min_val = float(min(np.min(y_true), np.min(y_pred)))
    max_val = float(max(np.max(y_true), np.max(y_pred)))
    ax.plot([min_val, max_val], [min_val, max_val], "r--", label="Perfect prediction")

    ax.set_xlabel("Actual toilet count")
    ax.set_ylabel("Predicted toilet count")
    ax.set_title(f"Actual vs Predicted (R²={r2:.3f})")
    ax.legend()
    plt.tight_layout()
    return fig


def plot_learning_curve(model: xgb.XGBRegressor, X_train: pd.DataFrame, y_train: pd.Series) -> plt.Figure:
    train_sizes, train_scores, test_scores = learning_curve(
        model,
        X_train,
        y_train,
        cv=5,
        train_sizes=np.linspace(0.1, 1.0, 10),
        scoring="r2",
        n_jobs=None,
    )

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(train_sizes, train_scores.mean(axis=1), "o-", label="Training score")
    ax.plot(train_sizes, test_scores.mean(axis=1), "o-", label="CV score")
    ax.set_xlabel("Training examples")
    ax.set_ylabel("R² score")
    ax.set_title("Learning Curve")
    ax.legend()
    plt.tight_layout()
    return fig


def log_metrics_to_postgres(run_id: str, experiment_name: str, metrics: Dict[str, float], params: Dict[str, Any]) -> None:
    conn = _connect()
    try:
        cur = conn.cursor()
        cur.execute(
            """
            INSERT INTO ml_runs (run_id, experiment_name, rmse, mae, r2,
                                model_version, n_estimators, max_depth, learning_rate, status)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (run_id) DO UPDATE SET
                experiment_name = EXCLUDED.experiment_name,
                rmse = EXCLUDED.rmse,
                mae = EXCLUDED.mae,
                r2 = EXCLUDED.r2,
                model_version = EXCLUDED.model_version,
                n_estimators = EXCLUDED.n_estimators,
                max_depth = EXCLUDED.max_depth,
                learning_rate = EXCLUDED.learning_rate,
                trained_at = CURRENT_TIMESTAMP,
                status = EXCLUDED.status
            """,
            (
                run_id,
                experiment_name,
                float(metrics.get("rmse", 0.0)),
                float(metrics.get("mae", 0.0)),
                float(metrics.get("r2", 0.0)),
                MODEL_VERSION,
                int(params.get("n_estimators", 0)),
                int(params.get("max_depth", 0)),
                float(params.get("learning_rate", 0.0)),
                "success",
            ),
        )
        conn.commit()
    finally:
        conn.close()


def train_model(
    experiment_name: Optional[str] = None,
    hyperparams: Optional[Dict[str, Any]] = None,
    use_parquet: bool = False,
) -> Tuple[xgb.XGBRegressor, str, Dict[str, float]]:
    """Обучение XGBoost модели с логированием в MLflow."""

    mlflow.set_tracking_uri(MLflowConfig.TRACKING_URI)
    exp_name = experiment_name or MLflowConfig.EXPERIMENT_NAME
    mlflow.set_experiment(exp_name)

    df = load_features_from_parquet() if use_parquet else load_features_from_postgres()

    print("\nFeature statistics:")
    print(get_feature_statistics(df))

    X, y, feature_cols = prepare_features(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=ModelConfig.random_state
    )

    params: Dict[str, Any] = asdict(ModelConfig()) if hyperparams is None else dict(hyperparams)

    # Conservative defaults for count regression
    params.setdefault("objective", "reg:squarederror")
    params.setdefault("n_jobs", -1)

    data_fp = _data_fingerprint(df)

    with mlflow.start_run() as run:
        run_id = run.info.run_id
        print(f"\nStarting MLflow run: {run_id}")

        mlflow.set_tag("data_fingerprint", data_fp)
        mlflow.set_tag("data_rows", int(len(df)))
        mlflow.set_tag("model_version", MODEL_VERSION)
        mlflow.set_tag("data_source", "parquet" if use_parquet else "postgres")

        model = xgb.XGBRegressor(**params)
        model.fit(X_train, y_train)

        y_pred_train = model.predict(X_train)
        y_pred = model.predict(X_test)

        metrics = {
            "rmse": float(np.sqrt(mean_squared_error(y_test, y_pred))),
            "mae": float(mean_absolute_error(y_test, y_pred)),
            "r2": float(r2_score(y_test, y_pred)),
            "train_rmse": float(np.sqrt(mean_squared_error(y_train, y_pred_train))),
            "train_r2": float(r2_score(y_train, y_pred_train)),
        }

        mlflow.log_params({k: v for k, v in params.items() if isinstance(v, (int, float, str, bool))})
        mlflow.log_metrics(metrics)

        mlflow.xgboost.log_model(model, MLflowConfig.ARTIFACT_PATH)

        fig1 = plot_feature_importance(model, feature_cols)
        mlflow.log_figure(fig1, "feature_importance.png")
        plt.close(fig1)

        fig2 = plot_actual_vs_predicted(y_test.to_numpy(), y_pred, metrics["r2"])
        mlflow.log_figure(fig2, "actual_vs_predicted.png")
        plt.close(fig2)

        # # learning_curve can be slow; keep it but simple
        # fig3 = plot_learning_curve(model, X_train, y_train)
        # mlflow.log_figure(fig3, "learning_curve.png")
        # plt.close(fig3)

        # Also store summary in Postgres for quick checks
        log_metrics_to_postgres(run_id, exp_name, metrics, params)

        print("\n✅ Experiment completed!")
        print(f"   RMSE: {metrics['rmse']:.3f}")
        print(f"   MAE:  {metrics['mae']:.3f}")
        print(f"   R²:   {metrics['r2']:.3f}")
        print(f"   Run ID: {run_id}")

        return model, run_id, metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Train toilet prediction model")
    parser.add_argument("--experiment", type=str, default=None, help="Experiment name")
    parser.add_argument("--parquet", action="store_true", help="Use Parquet files")
    args = parser.parse_args()

    train_model(experiment_name=args.experiment, use_parquet=args.parquet)


if __name__ == "__main__":
    main()
