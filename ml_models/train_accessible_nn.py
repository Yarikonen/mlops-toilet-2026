"""ml_models/train_accessible_nn.py

PyTorch MLP (бинарная классификация) для предсказания Accessible (True/False).
Логирование в MLflow: параметры, метрики по эпохам, графики.
Пишет сводные метрики в Postgres table `ml_accessible_runs`.

Запуск:
  poetry run python -m ml_models.train_accessible_nn
"""

from __future__ import annotations

import argparse
import os
import pickle
import tempfile
from typing import Any, Dict, Optional, Tuple

import matplotlib.pyplot as plt
import mlflow
import mlflow.pyfunc
import numpy as np
import pandas as pd
import psycopg2
import torch
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

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


def _device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _plot_learning_curves(history: Dict[str, list[float]]) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(history["train_loss"], label="train_loss")
    ax.plot(history["val_loss"], label="val_loss")
    ax2 = ax.twinx()
    ax2.plot(history["val_auc"], color="tab:green", label="val_auc", alpha=0.9)

    ax.set_xlabel("epoch")
    ax.set_title("Learning curves (loss + AUC)")
    ax.grid(True, alpha=0.3)

    # Merge legends
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc="best")

    plt.tight_layout()
    return fig


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


class MLPBinaryClassifier(nn.Module):
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)  # logits


class TorchAccessiblePyfuncModel(mlflow.pyfunc.PythonModel):
    """MLflow pyfunc wrapper: scaler + torch model, predict returns proba(Accessible=1)."""

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
        self._model: Optional[MLPBinaryClassifier] = None
        self._dev: Optional[torch.device] = None

    def load_context(self, context: mlflow.pyfunc.PythonModelContext) -> None:
        self._dev = _device()

        with open(context.artifacts["scaler"], "rb") as f:
            self._scaler = pickle.load(f)

        self._model = MLPBinaryClassifier(
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
            logits = self._model(x)
            proba = torch.sigmoid(logits).detach().cpu().numpy()

        return np.asarray(proba, dtype=np.float32)


def train_accessible_nn_model(
    experiment_name: Optional[str] = None,
    use_parquet: bool = False,
    epochs: int = 30,
    batch_size: int = 256,
    lr: float = 1e-3,
    weight_decay: float = 0.0,
    hidden_dim: int = 64,
    num_hidden_layers: int = 2,
    dropout: float = 0.1,
    seed: int = 42,
) -> Tuple[str, Dict[str, float]]:
    mlflow.set_tracking_uri(MLflowConfig.TRACKING_URI)
    exp_name = experiment_name or "toilet_accessible_classification"
    mlflow.set_experiment(exp_name)

    torch.manual_seed(seed)
    np.random.seed(seed)

    df = load_accessible_features_from_parquet() if use_parquet else load_accessible_features_from_postgres()

    print("\nFeature statistics:")
    print(get_accessible_feature_statistics(df))

    X_df, y_s, feature_cols = prepare_accessible_features(df)

    if y_s.nunique() < 2:
        raise RuntimeError("Target has <2 classes; cannot train classifier")

    X_train, X_test, y_train, y_test = train_test_split(
        X_df,
        y_s,
        test_size=0.2,
        random_state=seed,
        stratify=y_s,
    )

    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train,
        y_train,
        test_size=0.2,
        random_state=seed,
        stratify=y_train,
    )

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

    model = MLPBinaryClassifier(
        input_dim=X_tr_np.shape[1],
        hidden_dim=hidden_dim,
        num_hidden_layers=num_hidden_layers,
        dropout=dropout,
    ).to(dev)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    params: Dict[str, Any] = {
        "model": "torch_mlp_classifier",
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

    history: Dict[str, list[float]] = {"train_loss": [], "val_loss": [], "val_auc": []}

    with mlflow.start_run() as run:
        run_id = run.info.run_id
        print(f"\nStarting MLflow run: {run_id}")

        mlflow.set_tag("task", "accessible_classification")
        mlflow.set_tag("model_version", "torch_mlp_classifier")
        mlflow.set_tag("feature_cols", ",".join(feature_cols))
        mlflow.set_tag("data_rows", int(len(df)))

        mlflow.log_params(params)

        for epoch in range(epochs):
            model.train()
            train_losses: list[float] = []

            for xb, yb in train_loader:
                xb = xb.to(dev)
                yb = yb.to(dev)

                optimizer.zero_grad()
                logits = model(xb)
                loss = criterion(logits, yb)
                loss.backward()
                optimizer.step()
                train_losses.append(float(loss.detach().cpu().item()))

            train_loss = float(np.mean(train_losses)) if train_losses else 0.0

            model.eval()
            val_losses: list[float] = []
            pv_list: list[np.ndarray] = []
            yv_list: list[np.ndarray] = []

            with torch.no_grad():
                for xb, yb in val_loader:
                    xb = xb.to(dev)
                    yb = yb.to(dev)
                    logits = model(xb)
                    loss = criterion(logits, yb)
                    val_losses.append(float(loss.detach().cpu().item()))

                    proba = torch.sigmoid(logits).detach().cpu().numpy()
                    pv_list.append(proba)
                    yv_list.append(yb.detach().cpu().numpy())

            val_loss = float(np.mean(val_losses)) if val_losses else 0.0
            yv = np.concatenate(yv_list) if yv_list else y_val_np
            pv = np.concatenate(pv_list) if pv_list else np.zeros_like(yv)
            val_auc = float(roc_auc_score(yv, pv)) if len(np.unique(yv)) > 1 else 0.0

            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)
            history["val_auc"].append(val_auc)

            mlflow.log_metric("train_loss", train_loss, step=epoch)
            mlflow.log_metric("val_loss", val_loss, step=epoch)
            mlflow.log_metric("val_auc", val_auc, step=epoch)

        # Final evaluation on test
        model.eval()
        with torch.no_grad():
            logits_test = model(torch.from_numpy(X_test_np).to(dev))
            proba_test = torch.sigmoid(logits_test).detach().cpu().numpy()

        pred_test = (proba_test >= 0.5).astype(int)
        metrics = {
            "accuracy": float(accuracy_score(y_test_np, pred_test)),
            "f1": float(f1_score(y_test_np, pred_test, zero_division=0)),
            "roc_auc": float(roc_auc_score(y_test_np, proba_test)) if len(np.unique(y_test_np)) > 1 else 0.0,
        }
        mlflow.log_metrics(metrics)

        cm = confusion_matrix(y_test_np.astype(int), pred_test.astype(int), labels=[0, 1])
        fig_cm = _plot_confusion_matrix(cm)
        mlflow.log_figure(fig_cm, "confusion_matrix_nn.png")
        plt.close(fig_cm)

        fig_lc = _plot_learning_curves(history)
        mlflow.log_figure(fig_lc, "learning_curve_nn_cls.png")
        plt.close(fig_lc)

        # Log pyfunc model including scaler
        with tempfile.TemporaryDirectory() as tmp:
            state_path = os.path.join(tmp, "state_dict.pt")
            scaler_path = os.path.join(tmp, "scaler.pkl")

            torch.save(model.state_dict(), state_path)
            with open(scaler_path, "wb") as f:
                pickle.dump(scaler, f)

            pyfunc_model = TorchAccessiblePyfuncModel(
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

        log_accessible_metrics_to_postgres(run_id, exp_name, metrics, model_version="torch_mlp_classifier")

        print("\n✅ Accessible NN classifier completed!")
        print(f"   accuracy: {metrics['accuracy']:.3f}")
        print(f"   f1:       {metrics['f1']:.3f}")
        print(f"   roc_auc:  {metrics['roc_auc']:.3f}")
        print(f"   Run ID:   {run_id}")

        return run_id, metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Train torch MLP classifier for Accessible")
    parser.add_argument("--experiment", type=str, default=None, help="MLflow experiment name")
    parser.add_argument("--parquet", action="store_true", help="Use Parquet features")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--num-hidden-layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    train_accessible_nn_model(
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
