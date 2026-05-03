"""ml_models/predict.py

Инференс и анализ gaps (predicted - actual) по grid-ячейкам.
Сохраняет результаты в Postgres table `ml_predictions`.

Запуск локально:
  poetry run python -m ml_models.predict
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

from ml_models.config import DBConfig, MLflowConfig, MODEL_VERSION
from ml_models.features import load_features_from_postgres, prepare_features


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


def predict_gaps(model_uri: Optional[str] = None, experiment_name: Optional[str] = None) -> pd.DataFrame:
    """Предсказание и анализ gaps.

    Gap = predicted - actual
    Gap > 0: модель ожидает больше туалетов, чем есть (потенциальный дефицит)
    Gap < 0: потенциальный избыток
    """

    mlflow.set_tracking_uri(MLflowConfig.TRACKING_URI)

    exp_name = experiment_name or MLflowConfig.EXPERIMENT_NAME

    if model_uri is None:
        run_id = _latest_run_id(exp_name)
        model_uri = f"runs:/{run_id}/{MLflowConfig.ARTIFACT_PATH}"
        print(f"Using model from run: {run_id}")
    else:
        # Try to extract run_id from runs:/... URI
        run_id = model_uri.split("/")[1] if model_uri.startswith("runs:/") else "unknown"

    model = mlflow.pyfunc.load_model(model_uri)

    df = load_features_from_postgres()
    X, y, _ = prepare_features(df)

    preds = model.predict(X)
    preds = np.maximum(0, np.rint(preds)).astype(int)

    out = df[["lat_bin", "lon_bin"]].copy()
    out["predicted_count"] = preds
    out["actual_count"] = np.maximum(0, np.rint(y.to_numpy())).astype(int)
    out["gap"] = (out["predicted_count"] - out["actual_count"]).astype(int)

    # Save to Postgres
    conn = _connect()
    try:
        cur = conn.cursor()
        rows = [
            (
                int(r.lat_bin),
                int(r.lon_bin),
                int(r.predicted_count),
                int(r.actual_count),
                int(r.gap),
                MODEL_VERSION,
                str(run_id),
            )
            for r in out.itertuples(index=False)
        ]

        execute_values(
            cur,
            """
            INSERT INTO ml_predictions (
                lat_bin, lon_bin, predicted_count, actual_count, gap, model_version, run_id
            ) VALUES %s
            ON CONFLICT (lat_bin, lon_bin, run_id) DO UPDATE SET
                predicted_count = EXCLUDED.predicted_count,
                actual_count = EXCLUDED.actual_count,
                gap = EXCLUDED.gap,
                model_version = EXCLUDED.model_version,
                predicted_at = CURRENT_TIMESTAMP
            """,
            rows,
            page_size=1000,
        )
        conn.commit()
    finally:
        conn.close()

    print("\n" + "=" * 50)
    print("GAP ANALYSIS")
    print("=" * 50)
    print(f"Total grid cells: {len(out)}")
    print(f"Cells with gap > 0 (need more): {(out['gap'] > 0).sum()}")
    print(f"Cells with gap < 0 (oversupplied): {(out['gap'] < 0).sum()}")

    print("\n🔴 TOP-10 ZONES NEED MORE TOILETS:")
    print(out.nlargest(10, "gap").to_string(index=False))

    print("\n🟢 TOP-10 ZONES WITH EXCESS TOILETS:")
    print(out.nsmallest(10, "gap").to_string(index=False))

    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Predict toilet gaps")
    parser.add_argument("--model-uri", type=str, default=None, help="MLflow model URI")
    parser.add_argument("--experiment", type=str, default=None, help="MLflow experiment name")
    args = parser.parse_args()

    predict_gaps(model_uri=args.model_uri, experiment_name=args.experiment)


if __name__ == "__main__":
    main()
