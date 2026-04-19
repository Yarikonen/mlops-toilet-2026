from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.providers.apache.spark.operators.spark_submit import SparkSubmitOperator

from datetime import datetime
import requests


def ingest_data():
    import pandas as pd

    url = "https://data.gov.au/data/api/action/datastore_search"
    resource_id = "34076296-6692-4e30-b627-67b7c4eb1027"

    offset = 0
    limit = 1000
    all_rows = []

    while True:
        r = requests.get(url, params={
            "resource_id": resource_id,
            "limit": limit,
            "offset": offset
        }).json()

        rows = r["result"]["records"]
        if not rows:
            break

        all_rows.extend(rows)
        offset += limit

    df = pd.DataFrame(all_rows)
    df.to_csv("/opt/data/toilets.csv", index=False)

with DAG(
    dag_id="toilet_lr1_pipeline",
    start_date=datetime(2024, 1, 1),
    schedule="@weekly",
    catchup=False
) as dag:

    ingest = PythonOperator(
        task_id="ingest_api_data",
        python_callable=ingest_data
    )

    spark_job = SparkSubmitOperator(
        task_id="run_spark_job",
        application="/opt/spark_jobs/job.py",
        conn_id="spark_default",

        jars="/opt/spark/jars/postgresql-42.6.0.jar",

        executor_memory="2g",
        driver_memory="1g",

        conf={
            "spark.hadoop.fs.permissions.umask-mode": "000",
        },

        verbose=True,
    )
    

    ingest >> spark_job