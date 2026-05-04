"""airflow/dags/toilet_accessible_ml_pipeline.py

Новый ML pipeline (не заменяет старый): классификация доступности туалета (Accessible).

- extract_accessible_features: Spark job -> ml_accessible_features (Postgres + Parquet)
- train_accessible_xgb: XGBClassifier + MLflow + запись метрик в Postgres
- predict_accessible_xgb: предсказания -> ml_accessible_predictions
- train_accessible_nn: PyTorch MLP classifier + per-epoch MLflow metrics
- predict_accessible_nn: предсказания -> ml_accessible_predictions

Важно:
- Старый DAG `toilet_ml_pipeline` (регрессия по grid) остаётся без изменений.
"""

from __future__ import annotations

import sys
from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.operators.python import PythonOperator
from airflow.providers.apache.spark.operators.spark_submit import SparkSubmitOperator


def _ensure_imports() -> None:
    if "/opt" not in sys.path:
        sys.path.insert(0, "/opt")


def _train_xgb(**context):
    _ensure_imports()
    from ml_models.train_accessible_xgb import train_accessible_xgb_model

    dag_params = context.get("params", {})
    run_conf = (context.get("dag_run") and context["dag_run"].conf) or {}
    config = {**dag_params, **run_conf}

    # collect possible xgboost hyperparams from config
    hp_keys = (
        "n_estimators",
        "max_depth",
        "learning_rate",
        "subsample",
        "colsample_bytree",
        "booster",
        "rate_drop",
        "skip_drop",
        "random_state",
    )
    hyperparams: dict = {}
    for k in hp_keys:
        if k in config:
            hyperparams[k] = config[k]

    seed = int(config.get("seed", 42))
    use_parquet = bool(config.get("use_parquet", False))

    run_id, _ = train_accessible_xgb_model(
        experiment_name="toilet_accessible_airflow",
        use_parquet=use_parquet,
        seed=seed,
        hyperparams=hyperparams or None,
    )
    return run_id


def _predict_for_run(run_id: str):
    _ensure_imports()
    from ml_models.predict_accessible import predict_accessible

    predict_accessible(run_id=run_id, experiment_name="toilet_accessible_airflow")


def _predict_xgb(**context):
    run_id = context["ti"].xcom_pull(task_ids="train_accessible_xgb")
    if not run_id:
        raise RuntimeError("Missing run_id from train_accessible_xgb")
    _predict_for_run(str(run_id))


def _train_nn(**context):
    _ensure_imports()
    dag_params = context.get("params", {})
    run_conf = (context.get("dag_run") and context["dag_run"].conf) or {}
    config = {**dag_params, **run_conf}

    epochs = int(config.get("epochs", 30))
    batch_size = int(config.get("batch_size", 256))
    lr = float(config.get("lr", 1e-3))
    weight_decay = float(config.get("weight_decay", 0.0))
    hidden_dim = int(config.get("hidden_dim", 64))
    num_hidden_layers = int(config.get("num_hidden_layers", 2))
    dropout = float(config.get("dropout", 0.1))
    seed = int(config.get("seed", 42))
    use_parquet = bool(config.get("use_parquet", False))

    from ml_models.train_accessible_nn import train_accessible_nn_model

    run_id, _ = train_accessible_nn_model(
        experiment_name="toilet_accessible_airflow",
        use_parquet=use_parquet,
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        weight_decay=weight_decay,
        hidden_dim=hidden_dim,
        num_hidden_layers=num_hidden_layers,
        dropout=dropout,
        seed=seed,
    )
    return run_id


def _predict_nn(**context):
    run_id = context["ti"].xcom_pull(task_ids="train_accessible_nn")
    if not run_id:
        raise RuntimeError("Missing run_id from train_accessible_nn")
    _predict_for_run(str(run_id))


with DAG(
    dag_id="toilet_accessible_ml_pipeline",
    start_date=datetime(2024, 1, 1),
    schedule=None,
    catchup=False,
    default_args={"retries": 1, "retry_delay": timedelta(minutes=5)},
    doc_md=__doc__,
    params={
        "epochs": 30,
        "batch_size": 256,
        "lr": 1e-3,
        "weight_decay": 0.0,
        "hidden_dim": 64,
        "num_hidden_layers": 2,
        "dropout": 0.1,
        "use_parquet": False,
        "seed": 42,
        # xgb defaults (can be overridden)
        "n_estimators": 200,
        "max_depth": 5,
        "learning_rate": 0.1,
        "subsample": 0.9,
        "colsample_bytree": 0.9,
        # hype override can be provided as dict under `hype_hyperparams`
        "hype_hyperparams": None,
    },
) as dag:
    extract_accessible_features = SparkSubmitOperator(
        task_id="extract_accessible_features",
        application="/opt/spark_jobs/accessible_features.py",
        conn_id="spark_default",
        jars="/opt/spark/jars/postgresql-42.6.0.jar",
        executor_memory="2g",
        driver_memory="1g",
        conf={
            "spark.hadoop.fs.permissions.umask-mode": "000",
        },
        verbose=True,
    )

    train_accessible_xgb = PythonOperator(
        task_id="train_accessible_xgb",
        python_callable=_train_xgb,
    )

    predict_accessible_xgb = PythonOperator(
        task_id="predict_accessible_xgb",
        python_callable=_predict_xgb,
    )

    train_accessible_hype = PythonOperator(
        task_id="train_accessible_hype",
        python_callable=_train_hype,
    )

    predict_accessible_hype = PythonOperator(
        task_id="predict_accessible_hype",
        python_callable=_predict_hype,
    )

    train_accessible_nn = PythonOperator(
        task_id="train_accessible_nn",
        python_callable=_train_nn,
    )

    predict_accessible_nn = PythonOperator(
        task_id="predict_accessible_nn",
        python_callable=_predict_nn,
    )

    notify = BashOperator(
        task_id="notify_complete",
        bash_command="echo 'Accessible ML pipeline completed successfully!'",
    )

    (
        extract_accessible_features
        >> train_accessible_xgb
        >> predict_accessible_xgb
        >> train_accessible_hype
        >> predict_accessible_hype
        >> train_accessible_nn
        >> predict_accessible_nn
        >> notify
    )
