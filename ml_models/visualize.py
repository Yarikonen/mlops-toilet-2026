"""ml_models/visualize.py

Визуализация результатов (графики для отчёта):
- распределение gap
- actual vs predicted
- сравнение экспериментов MLflow

Запуск:
  poetry run python -m ml_models.visualize
"""

from __future__ import annotations

import os

import matplotlib.pyplot as plt
import mlflow
import pandas as pd
import psycopg2

from ml_models.config import DBConfig, MLflowConfig


def _connect():
    return psycopg2.connect(
        host=DBConfig.HOST,
        port=DBConfig.PORT,
        database=DBConfig.NAME,
        user=DBConfig.USER,
        password=DBConfig.PASSWORD,
    )


def plot_gap_distribution(output_path: str = "/opt/data/processed/gap_analysis.png") -> None:
    conn = _connect()
    try:
        query = """
        SELECT gap, predicted_count, actual_count
        FROM ml_predictions
        WHERE run_id = (
            SELECT run_id FROM ml_predictions ORDER BY predicted_at DESC LIMIT 1
        )
        """
        df = pd.read_sql_query(query, conn)
    finally:
        conn.close()

    if len(df) == 0:
        print("No prediction rows found in ml_predictions")
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].hist(df["gap"], bins=50, edgecolor="black", alpha=0.7)
    axes[0].axvline(x=0, color="r", linestyle="--", label="Zero gap")
    axes[0].set_xlabel("Gap (predicted - actual)")
    axes[0].set_ylabel("Count")
    axes[0].set_title("Distribution of Gaps")
    axes[0].legend()

    axes[1].scatter(df["actual_count"], df["predicted_count"], alpha=0.5)
    min_val = min(df["actual_count"].min(), df["predicted_count"].min())
    max_val = max(df["actual_count"].max(), df["predicted_count"].max())
    axes[1].plot([min_val, max_val], [min_val, max_val], "r--")
    axes[1].set_xlabel("Actual count")
    axes[1].set_ylabel("Predicted count")
    axes[1].set_title("Actual vs Predicted")

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=150)
    plt.close(fig)

    print(f"Saved: {output_path}")


def plot_mlflow_comparison(output_path: str = "/opt/data/processed/mlflow_comparison.png") -> None:
    mlflow.set_tracking_uri(MLflowConfig.TRACKING_URI)

    runs = mlflow.search_runs(
        experiment_names=[MLflowConfig.EXPERIMENT_NAME],
        order_by=["start_time ASC"],
    )

    if runs is None or len(runs) == 0:
        print("No experiments found in MLflow")
        return

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    axes[0].plot(range(len(runs)), runs["metrics.r2"], "o-")
    axes[0].set_xlabel("Run #")
    axes[0].set_ylabel("R²")
    axes[0].set_title("R² by Run")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(range(len(runs)), runs["metrics.rmse"], "o-", color="orange")
    axes[1].set_xlabel("Run #")
    axes[1].set_ylabel("RMSE")
    axes[1].set_title("RMSE by Run")
    axes[1].grid(True, alpha=0.3)

    if "params.max_depth" in runs.columns:
        axes[2].scatter(runs["params.max_depth"].astype(float), runs["metrics.r2"], alpha=0.7)
        axes[2].set_xlabel("max_depth")
        axes[2].set_ylabel("R²")
        axes[2].set_title("max_depth vs R²")
        axes[2].grid(True, alpha=0.3)
    else:
        axes[2].axis("off")

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=150)
    plt.close(fig)

    print(f"Saved: {output_path}")


def main() -> None:
    plot_gap_distribution()
    plot_mlflow_comparison()


if __name__ == "__main__":
    main()
