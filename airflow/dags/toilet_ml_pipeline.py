"""airflow/dags/toilet_ml_pipeline.py

ML Pipeline для Lab 2 (строго по плану):
- extract_features: Spark job -> ml_features (Postgres + Parquet)
- train_model: XGBoost + MLflow + запись метрик в Postgres
- predict_gaps: предсказания + gap analysis -> ml_predictions
- visualize: сохранение графиков в /opt/data/processed
"""

from __future__ import annotations

import sys
from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.operators.python import PythonOperator
from airflow.providers.apache.spark.operators.spark_submit import SparkSubmitOperator



def _ensure_imports() -> None:
    # ml_models is mounted into /opt/ml_models; package import needs /opt on sys.path
    if "/opt" not in sys.path:
        sys.path.insert(0, "/opt")


def _train(**context) -> None:
    _ensure_imports()
    from ml_models.train_nn import train_nn_model

    dag_params = context.get("params", {})
    run_conf = (context.get("dag_run") and context["dag_run"].conf) or {}

    config = {**dag_params, **run_conf}

    epochs = config.get("epochs", 20)

    train_nn_model(
        experiment_name="toilet_location_airflow",
        use_parquet=False,
        epochs=epochs,
    )


def _predict() -> None:
    _ensure_imports()
    from ml_models.predict import predict_gaps

    predict_gaps(experiment_name="toilet_location_airflow")


def _visualize() -> None:
    _ensure_imports()
    from ml_models.visualize import plot_gap_distribution, plot_mlflow_comparison

    plot_gap_distribution()
    plot_mlflow_comparison()


with DAG(
    dag_id="toilet_ml_pipeline",
    start_date=datetime(2024, 1, 1),
    schedule="@monthly",
    catchup=False,
    default_args={"retries": 1, "retry_delay": timedelta(minutes=5)},
    doc_md=__doc__,
    params={"epochs": 20},
) as dag:
    extract_features = SparkSubmitOperator(
        task_id="extract_features",
        application="/opt/spark_jobs/ml_features.py",
        conn_id="spark_default",
        jars="/opt/spark/jars/postgresql-42.6.0.jar",
        executor_memory="2g",
        driver_memory="1g",
        conf={
            "spark.hadoop.fs.permissions.umask-mode": "000",
        },
        verbose=True,
    )

    train_model_task = PythonOperator(
        task_id="train_model",
        python_callable=_train,
    )

    predict_gaps_task = PythonOperator(
        task_id="predict_gaps",
        python_callable=_predict,
    )

    visualize_task = PythonOperator(
        task_id="visualize",
        python_callable=_visualize,
    )

    notify = BashOperator(
        task_id="notify_complete",
        bash_command="echo 'ML Pipeline completed successfully!'",
    )

    extract_features >> train_model_task >> predict_gaps_task >> visualize_task >> notify
