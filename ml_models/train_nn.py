"""ml_models/train_nn.py

PyTorch MLP (регрессия) для задачи прогнозирования количества туалетов по grid-ячейкам.

Требования из плана:
- несколько гипотез: baseline XGBoost уже есть, этот файл добавляет NN-эксперимент
- трекинг экспериментов через MLflow (параметры, метрики, кривые обучения)

Важно:
- для MLP нужен scaling признаков, поэтому scaler логируется вместе с моделью
- модель логируется в MLflow как pyfunc (с предобработкой), чтобы predict мог брать
  «последнюю модель из трекера» независимо от алгоритма.

Запуск:
  poetry run python -m ml_models.train_nn
"""

from __future__ import annotations

import argparse
import os
import pickle
import tempfile
from dataclasses import asdict
from typing import Any, Dict, Optional, Tuple

import matplotlib.pyplot as plt
import mlflow
import mlflow.pyfunc
import numpy as np
import pandas as pd
import psycopg2
import torch
from pandas.util import hash_pandas_object
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

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
    cols = [c for c in df.columns if c not in {"created_at"}]
    hashed = hash_pandas_object(df[cols], index=False).values
    return hex(int(hashed.sum()) & 0xFFFFFFFFFFFF)


def _device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


class MLPRegressor(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
        num_hidden_layers: int = 2,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()

        layers: list[nn.Module] = []
        in_dim = input_dim
        for _ in range(num_hidden_layers):
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.ReLU())
            if dropout and dropout > 0:
                layers.append(nn.Dropout(dropout))
            in_dim = hidden_dim

        layers.append(nn.Linear(in_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # (N, D) -> (N,)
        return self.net(x).squeeze(-1)


class TorchMLPPyfuncModel(mlflow.pyfunc.PythonModel):
    """MLflow pyfunc wrapper: включает scaler + torch model."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_hidden_layers: int,
        dropout: float,
    ) -> None:
        self.input_dim = int(input_dim)
        self.hidden_dim = int(hidden_dim)
        self.num_hidden_layers = int(num_hidden_layers)
        self.dropout = float(dropout)

        self._scaler: Optional[StandardScaler] = None
        self._model: Optional[MLPRegressor] = None
        self._dev: Optional[torch.device] = None

    def load_context(self, context: mlflow.pyfunc.PythonModelContext) -> None:
        self._dev = _device()

        with open(context.artifacts["scaler"], "rb") as f:
            self._scaler = pickle.load(f)

        self._model = MLPRegressor(
            input_dim=self.input_dim,
            hidden_dim=self.hidden_dim,
            num_hidden_layers=self.num_hidden_layers,
            dropout=self.dropout,
        )

        state = torch.load(context.artifacts["state_dict"], map_location="cpu")
        self._model.load_state_dict(state)
        self._model.to(self._dev)
        self._model.eval()

    def predict(self, context: mlflow.pyfunc.PythonModelContext, model_input):
        if self._model is None or self._scaler is None or self._dev is None:
            raise RuntimeError("Model is not loaded")

        if isinstance(model_input, pd.DataFrame):
            x_np = model_input.to_numpy(dtype=np.float32, copy=False)
        else:
            x_np = np.asarray(model_input, dtype=np.float32)

        x_np = self._scaler.transform(x_np).astype(np.float32, copy=False)
        x = torch.from_numpy(x_np).to(self._dev)

        with torch.no_grad():
            preds = self._model(x).detach().cpu().numpy()

        return preds


def _plot_learning_curves(history: Dict[str, list[float]]) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(history["train_mse"], label="train_mse")
    ax.plot(history["val_rmse"], label="val_rmse")
    ax.set_xlabel("epoch")
    ax.set_title("Learning curves")
    ax.grid(True, alpha=0.3)
    ax.legend()
    plt.tight_layout()
    return fig


def _plot_actual_vs_predicted(y_true: np.ndarray, y_pred: np.ndarray, r2: float) -> plt.Figure:
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


def log_metrics_to_postgres_nn(run_id: str, experiment_name: str, metrics: Dict[str, float], params: Dict[str, Any]) -> None:
    """Пишем в ml_runs, но параметры маппим на существующие колонки (без изменения схемы)."""

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
                "torch_mlp",
                int(params.get("epochs", 0)),
                int(params.get("hidden_dim", 0)),
                float(params.get("lr", 0.0)),
                "success",
            ),
        )
        conn.commit()
    finally:
        conn.close()


def train_nn_model(
    experiment_name: Optional[str] = None,
    use_parquet: bool = False,
    epochs: int = 50,
    batch_size: int = 128,
    lr: float = 1e-3,
    weight_decay: float = 0.0,
    hidden_dim: int = 64,
    num_hidden_layers: int = 2,
    dropout: float = 0.0,
    seed: int = 42,
) -> Tuple[nn.Module, str, Dict[str, float]]:
    mlflow.set_tracking_uri(MLflowConfig.TRACKING_URI)
    exp_name = experiment_name or MLflowConfig.EXPERIMENT_NAME
    mlflow.set_experiment(exp_name)

    torch.manual_seed(seed)
    np.random.seed(seed)

    df = load_features_from_parquet() if use_parquet else load_features_from_postgres()

    print("\nFeature statistics:")
    print(get_feature_statistics(df))

    X_df, y_s, feature_cols = prepare_features(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X_df, y_s, test_size=0.2, random_state=seed
    )

    # Inner validation split for per-epoch logging
    X_tr, X_val, y_tr, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=seed)

    scaler = StandardScaler()
    X_tr_np = scaler.fit_transform(X_tr.to_numpy(dtype=np.float32))
    X_val_np = scaler.transform(X_val.to_numpy(dtype=np.float32))
    X_test_np = scaler.transform(X_test.to_numpy(dtype=np.float32))

    y_tr_np = y_tr.to_numpy(dtype=np.float32)
    y_val_np = y_val.to_numpy(dtype=np.float32)
    y_test_np = y_test.to_numpy(dtype=np.float32)

    train_ds = TensorDataset(torch.from_numpy(X_tr_np), torch.from_numpy(y_tr_np))
    val_ds = TensorDataset(torch.from_numpy(X_val_np), torch.from_numpy(y_val_np))

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    dev = _device()

    model = MLPRegressor(
        input_dim=X_tr_np.shape[1],
        hidden_dim=hidden_dim,
        num_hidden_layers=num_hidden_layers,
        dropout=dropout,
    ).to(dev)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    data_fp = _data_fingerprint(df)

    params: Dict[str, Any] = {
        "model": "torch_mlp",
        "epochs": int(epochs),
        "batch_size": int(batch_size),
        "lr": float(lr),
        "weight_decay": float(weight_decay),
        "hidden_dim": int(hidden_dim),
        "num_hidden_layers": int(num_hidden_layers),
        "dropout": float(dropout),
        "seed": int(seed),
        "data_source": "parquet" if use_parquet else "postgres",
    }

    history: Dict[str, list[float]] = {"train_mse": [], "val_rmse": []}

    with mlflow.start_run() as run:
        run_id = run.info.run_id
        print(f"\nStarting MLflow run: {run_id}")

        mlflow.set_tag("data_fingerprint", data_fp)
        mlflow.set_tag("data_rows", int(len(df)))
        mlflow.set_tag("model_version", "torch_mlp")
        mlflow.set_tag("feature_cols", ",".join(feature_cols))

        mlflow.log_params(params)

        for epoch in range(epochs):
            model.train()
            train_losses: list[float] = []

            for xb, yb in train_loader:
                xb = xb.to(dev)
                yb = yb.to(dev)

                optimizer.zero_grad()
                pred = model(xb)
                loss = criterion(pred, yb)
                loss.backward()
                optimizer.step()
                train_losses.append(float(loss.detach().cpu().item()))

            train_mse = float(np.mean(train_losses)) if train_losses else 0.0

            # Validation
            model.eval()
            preds_val: list[np.ndarray] = []
            y_val_list: list[np.ndarray] = []
            with torch.no_grad():
                for xb, yb in val_loader:
                    xb = xb.to(dev)
                    pred = model(xb).detach().cpu().numpy()
                    preds_val.append(pred)
                    y_val_list.append(yb.numpy())

            yv = np.concatenate(y_val_list) if y_val_list else y_val_np
            pv = np.concatenate(preds_val) if preds_val else np.zeros_like(yv)

            val_rmse = float(np.sqrt(mean_squared_error(yv, pv)))

            history["train_mse"].append(train_mse)
            history["val_rmse"].append(val_rmse)

            mlflow.log_metric("train_mse", train_mse, step=epoch)
            mlflow.log_metric("val_rmse", val_rmse, step=epoch)

        # Final evaluation on test
        model.eval()
        with torch.no_grad():
            test_preds = model(torch.from_numpy(X_test_np).to(dev)).detach().cpu().numpy()

        metrics = {
            "rmse": float(np.sqrt(mean_squared_error(y_test_np, test_preds))),
            "mae": float(mean_absolute_error(y_test_np, test_preds)),
            "r2": float(r2_score(y_test_np, test_preds)),
        }
        mlflow.log_metrics(metrics)

        # Plots
        fig1 = _plot_learning_curves(history)
        mlflow.log_figure(fig1, "learning_curve_nn.png")
        plt.close(fig1)

        fig2 = _plot_actual_vs_predicted(y_test_np, test_preds, metrics["r2"])
        mlflow.log_figure(fig2, "actual_vs_predicted_nn.png")
        plt.close(fig2)

        # Log pyfunc model including scaler
        with tempfile.TemporaryDirectory() as tmp:
            state_path = os.path.join(tmp, "state_dict.pt")
            scaler_path = os.path.join(tmp, "scaler.pkl")

            torch.save(model.state_dict(), state_path)
            with open(scaler_path, "wb") as f:
                pickle.dump(scaler, f)

            pyfunc_model = TorchMLPPyfuncModel(
                input_dim=X_tr_np.shape[1],
                hidden_dim=hidden_dim,
                num_hidden_layers=num_hidden_layers,
                dropout=dropout,
            )

            mlflow.pyfunc.log_model(
                artifact_path=MLflowConfig.ARTIFACT_PATH,
                python_model=pyfunc_model,
                artifacts={"state_dict": state_path, "scaler": scaler_path},
                pip_requirements=[
                    "mlflow",
                    "numpy",
                    "pandas",
                    "scikit-learn",
                    "torch",
                ],
            )

        # Also store summary in Postgres for quick checks
        log_metrics_to_postgres_nn(run_id, exp_name, metrics, params)

        print("\n✅ NN Experiment completed!")
        print(f"   RMSE: {metrics['rmse']:.3f}")
        print(f"   MAE:  {metrics['mae']:.3f}")
        print(f"   R²:   {metrics['r2']:.3f}")
        print(f"   Run ID: {run_id}")

        return model, run_id, metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Train torch MLP model")
    parser.add_argument("--experiment", type=str, default=None, help="Experiment name")
    parser.add_argument("--parquet", action="store_true", help="Use Parquet files")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--num-hidden-layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    train_nn_model(
        experiment_name=args.experiment,
        use_parquet=args.parquet,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        hidden_dim=args.hidden_dim,
        num_hidden_layers=args.num_hidden_layers,
        dropout=args.dropout,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
