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

    conf = context["dag_run"].conf or {}


    run_id, _ = train_accessible_xgb_model(experiment_name="toilet_accessible_airflow", use_parquet=False)
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
    epochs = context["dag_run"].conf.get("epochs", 20)

    from ml_models.train_accessible_nn import train_accessible_nn_model

    run_id, _ = train_accessible_nn_model(experiment_name="toilet_accessible_airflow", use_parquet=False, epochs=epochs)
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
        >> train_accessible_nn
        >> predict_accessible_nn
        >> notify
    )
