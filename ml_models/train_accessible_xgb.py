"""ml_models/train_accessible_xgb.py

Обучение XGBClassifier для задачи бинарной классификации доступности туалета.
Логирование в MLflow: параметры, метрики, confusion matrix.
Пишет сводные метрики в Postgres table `ml_accessible_runs`.

Запуск:
  poetry run python -m ml_models.train_accessible_xgb
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
from typing import Any, Dict, Optional, Tuple

import matplotlib.pyplot as plt
import mlflow
import mlflow.pyfunc
import numpy as np
import pandas as pd
import psycopg2
import xgboost as xgb
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

from ml_models.config import DBConfig, MLflowConfig
from ml_models.accessible_features import (
    get_accessible_feature_statistics,
    load_accessible_features_from_parquet,
    load_accessible_features_from_postgres,
    prepare_accessible_features,
)


def _connect():
    return psycopg2.connect(
        host=DBConfig.HOST,
        port=DBConfig.PORT,
        database=DBConfig.NAME,
        user=DBConfig.USER,
        password=DBConfig.PASSWORD,
    )


def _plot_confusion_matrix(cm: np.ndarray) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.imshow(cm, cmap="Blues")
    ax.set_title("Confusion Matrix")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    for (i, j), v in np.ndenumerate(cm):
        ax.text(j, i, str(v), ha="center", va="center")
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["0", "1"])
    ax.set_yticklabels(["0", "1"])
    plt.tight_layout()
    return fig


def log_accessible_metrics_to_postgres(
    run_id: str,
    experiment_name: str,
    metrics: Dict[str, float],
    model_version: str,
) -> None:
    conn = _connect()
    try:
        cur = conn.cursor()
        cur.execute(
            """
            INSERT INTO ml_accessible_runs (run_id, experiment_name, accuracy, f1, roc_auc, model_version, status)
            VALUES (%s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (run_id) DO UPDATE SET
                experiment_name = EXCLUDED.experiment_name,
                accuracy = EXCLUDED.accuracy,
                f1 = EXCLUDED.f1,
                roc_auc = EXCLUDED.roc_auc,
                model_version = EXCLUDED.model_version,
                trained_at = CURRENT_TIMESTAMP,
                status = EXCLUDED.status
            """,
            (
                run_id,
                experiment_name,
                float(metrics.get("accuracy", 0.0)),
                float(metrics.get("f1", 0.0)),
                float(metrics.get("roc_auc", 0.0)),
                model_version,
                "success",
            ),
        )
        conn.commit()
    finally:
        conn.close()


class XGBAccessiblePyfuncModel(mlflow.pyfunc.PythonModel):
    """pyfunc wrapper to expose predict_proba as a stable interface."""

    def load_context(self, context: mlflow.pyfunc.PythonModelContext) -> None:
        self._booster = xgb.Booster()
        self._booster.load_model(context.artifacts["booster"])

    def predict(self, context: mlflow.pyfunc.PythonModelContext, model_input):
        if isinstance(model_input, pd.DataFrame):
            X_df = model_input
            # Preserve feature names to satisfy XGBoost validation.
            dmat = xgb.DMatrix(
                X_df.astype(np.float32, copy=False),
                feature_names=[str(c) for c in X_df.columns],
            )
            proba_pos = self._booster.predict(dmat)
        else:
            X = np.asarray(model_input, dtype=np.float32)
            dmat = xgb.DMatrix(X)
            # Numpy input has no column names; skip strict validation.
            proba_pos = self._booster.predict(dmat, validate_features=False)
        return np.asarray(proba_pos, dtype=np.float32)


def train_accessible_xgb_model(
    experiment_name: Optional[str] = None,
    use_parquet: bool = False,
    seed: int = 42,
    hyperparams: Optional[Dict[str, Any]] = None,
) -> Tuple[str, Dict[str, float]]:
    mlflow.set_tracking_uri(MLflowConfig.TRACKING_URI)
    exp_name = experiment_name or "toilet_accessible_classification"
    mlflow.set_experiment(exp_name)

    df = load_accessible_features_from_parquet() if use_parquet else load_accessible_features_from_postgres()

    print("\nFeature statistics:")
    print(get_accessible_feature_statistics(df))

    X, y, feature_cols = prepare_accessible_features(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=seed,
        stratify=y if y.nunique() > 1 else None,
    )

    params: Dict[str, Any] = {} if hyperparams is None else dict(hyperparams)

    # Stable binary classification defaults
    params.setdefault("n_estimators", 200)
    params.setdefault("max_depth", 5)
    params.setdefault("learning_rate", 0.1)
    params.setdefault("subsample", 0.9)
    params.setdefault("colsample_bytree", 0.9)
    params.setdefault("random_state", seed)
    params.setdefault("objective", "binary:logistic")
    params.setdefault("n_jobs", -1)

    model = xgb.XGBClassifier(**params)

    with mlflow.start_run() as run:
        run_id = run.info.run_id
        print(f"\nStarting MLflow run: {run_id}")

        mlflow.set_tag("task", "accessible_classification")
        mlflow.set_tag("model_version", "xgb_classifier")
        mlflow.set_tag("feature_cols", ",".join(feature_cols))
        mlflow.set_tag("data_rows", int(len(df)))

        mlflow.log_params({k: v for k, v in params.items() if isinstance(v, (int, float, str, bool))})

        model.fit(X_train, y_train)

        proba_test = model.predict_proba(X_test)[:, 1]
        pred_test = (proba_test >= 0.5).astype(int)

        metrics = {
            "accuracy": float(accuracy_score(y_test, pred_test)),
            "f1": float(f1_score(y_test, pred_test, zero_division=0)),
            "roc_auc": float(roc_auc_score(y_test, proba_test)) if y_test.nunique() > 1 else 0.0,
        }
        mlflow.log_metrics(metrics)

        cm = confusion_matrix(y_test, pred_test, labels=[0, 1])
        fig = _plot_confusion_matrix(cm)
        mlflow.log_figure(fig, "confusion_matrix_xgb.png")
        plt.close(fig)

        # Log as pyfunc (predict returns proba of class=1)
        import tempfile
        import os

        with tempfile.TemporaryDirectory() as tmp:
            booster_path = os.path.join(tmp, "model.json")
            model.get_booster().save_model(booster_path)

            mlflow.pyfunc.log_model(
                artifact_path=MLflowConfig.ARTIFACT_PATH,
                python_model=XGBAccessiblePyfuncModel(),
                artifacts={"booster": booster_path},
                pip_requirements=[
                    "mlflow",
                    "numpy",
                    "pandas",
                    "xgboost",
                ],
            )

        log_accessible_metrics_to_postgres(run_id, exp_name, metrics, model_version="xgb_classifier")

        print("\n✅ Accessible XGBClassifier completed!")
        print(f"   accuracy: {metrics['accuracy']:.3f}")
        print(f"   f1:       {metrics['f1']:.3f}")
        print(f"   roc_auc:  {metrics['roc_auc']:.3f}")
        print(f"   Run ID:   {run_id}")

        return run_id, metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Train XGBClassifier for accessible classification")
    parser.add_argument("--experiment", type=str, default=None, help="MLflow experiment name")
    parser.add_argument("--parquet", action="store_true", help="Use Parquet features")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    train_accessible_xgb_model(experiment_name=args.experiment, use_parquet=args.parquet, seed=args.seed)


if __name__ == "__main__":
    main()
