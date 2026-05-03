"""ml_models/predict_accessible.py

Инференс для задачи Accessible (0/1) на датасете ml_accessible_features.
Сохраняет результаты в Postgres table `ml_accessible_predictions`.

Модель берётся из MLflow по run_id (или последнему run в experiment).
Ожидается, что модель залогирована как MLflow pyfunc и `predict(X)` возвращает
вероятность класса 1 (Accessible=True).

Запуск:
  poetry run python -m ml_models.predict_accessible
  poetry run python -m ml_models.predict_accessible --run-id <run_id>
"""

from __future__ import annotations

import argparse
from typing import Optional

import mlflow
import mlflow.pyfunc
import numpy as np
import pandas as pd
import psycopg2
from psycopg2.extras import execute_values

from ml_models.config import DBConfig, MLflowConfig
from ml_models.accessible_features import load_accessible_features_from_postgres, prepare_accessible_features


def _connect():
    return psycopg2.connect(
        host=DBConfig.HOST,
        port=DBConfig.PORT,
        database=DBConfig.NAME,
        user=DBConfig.USER,
        password=DBConfig.PASSWORD,
    )


def _latest_run_id(experiment_name: str) -> str:
    mlflow.set_tracking_uri(MLflowConfig.TRACKING_URI)
    mlflow.set_experiment(experiment_name)

    runs = mlflow.search_runs(
        experiment_names=[experiment_name],
        order_by=["start_time DESC"],
        max_results=1,
    )
    if runs is None or len(runs) == 0:
        raise RuntimeError(f"No MLflow runs found in experiment: {experiment_name}")

    return str(runs.iloc[0]["run_id"])


def predict_accessible(
    run_id: Optional[str] = None,
    model_uri: Optional[str] = None,
    experiment_name: Optional[str] = None,
    threshold: float = 0.5,
) -> pd.DataFrame:
    mlflow.set_tracking_uri(MLflowConfig.TRACKING_URI)

    exp_name = experiment_name or "toilet_accessible_classification"

    if model_uri is None:
        if run_id is None:
            run_id = _latest_run_id(exp_name)
        model_uri = f"runs:/{run_id}/{MLflowConfig.ARTIFACT_PATH}"
        print(f"Using model from run: {run_id}")
    else:
        run_id = run_id or (model_uri.split("/")[1] if model_uri.startswith("runs:/") else "unknown")

    model = mlflow.pyfunc.load_model(model_uri)

    df = load_accessible_features_from_postgres()
    X, y, _ = prepare_accessible_features(df)

    proba = model.predict(X)
    proba = np.asarray(proba, dtype=np.float32).reshape(-1)
    pred = (proba >= float(threshold)).astype(int)

    out = df[["_id"]].copy()
    out["predicted_accessible"] = pred
    out["predicted_proba"] = proba
    out["actual_accessible"] = y.to_numpy(dtype=int)

    # Best-effort model_version from MLflow tags
    model_version = "unknown"
    try:
        info = mlflow.get_run(str(run_id)).data
        model_version = info.tags.get("model_version", info.tags.get("mlflow.runName", "unknown"))
    except Exception:
        pass

    conn = _connect()
    try:
        cur = conn.cursor()
        rows = [
            (
                str(toilet_id),
                int(predicted_accessible),
                float(predicted_proba),
                int(actual_accessible),
                str(model_version),
                str(run_id),
            )
            for toilet_id, predicted_accessible, predicted_proba, actual_accessible in out[
                ["_id", "predicted_accessible", "predicted_proba", "actual_accessible"]
            ].itertuples(index=False, name=None)
        ]

        execute_values(
            cur,
            """
            INSERT INTO ml_accessible_predictions (
                _id, predicted_accessible, predicted_proba, actual_accessible, model_version, run_id
            ) VALUES %s
            ON CONFLICT (_id, run_id) DO UPDATE SET
                predicted_accessible = EXCLUDED.predicted_accessible,
                predicted_proba = EXCLUDED.predicted_proba,
                actual_accessible = EXCLUDED.actual_accessible,
                model_version = EXCLUDED.model_version,
                predicted_at = CURRENT_TIMESTAMP
            """,
            rows,
            page_size=2000,
        )
        conn.commit()
    finally:
        conn.close()

    print("\n" + "=" * 50)
    print("ACCESSIBLE PREDICTION")
    print("=" * 50)
    print(f"Total toilets: {len(out)}")
    print(f"Predicted accessible rate: {out['predicted_accessible'].mean():.3f}")

    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Predict Accessible with latest MLflow model")
    parser.add_argument("--run-id", type=str, default=None, help="MLflow run id")
    parser.add_argument("--model-uri", type=str, default=None, help="MLflow model uri")
    parser.add_argument("--experiment", type=str, default=None, help="MLflow experiment name")
    parser.add_argument("--threshold", type=float, default=0.5)
    args = parser.parse_args()

    predict_accessible(
        run_id=args.run_id,
        model_uri=args.model_uri,
        experiment_name=args.experiment,
        threshold=args.threshold,
    )


if __name__ == "__main__":
    main()
