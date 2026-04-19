from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.operators.python import PythonOperator
from airflow.providers.apache.spark.operators.spark_submit import SparkSubmitOperator
from airflow.providers.http.operators.http import SimpleHttpOperator

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


def submit_spark_job():
    import subprocess
    import time
    
    cmd = [
        "docker", "exec", "mlops-spark-master",
        "spark-submit",
        "--master", "spark://spark-master:7077",
        "--conf", "spark.executor.memory=1g",
        "--conf", "spark.driver.memory=1g",
        "--jars", "/opt/bitnami/spark/jars/postgresql-42.6.0.jar",
        "/opt/spark_jobs/job.py"
    ]
    
    print(f"Command: {' '.join(cmd)}")
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode == 0:
        print(result.stdout)
    else:
        print(result.stderr)
        raise Exception(f"Spark job failed with code {result.returncode}")
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
        application="/opt/airflow/jobs/toilet_job.py", 
        conn_id="spark_default2",

        conf={
            "spark.master": "spark://spark-master:7077",
        },

        packages="org.postgresql:postgresql:42.7.3",

        executor_memory="2g",
        driver_memory="1g",

        verbose=True,
    )
    

    ingest >> spark_job