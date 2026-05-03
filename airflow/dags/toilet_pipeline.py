from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.providers.apache.spark.operators.spark_submit import SparkSubmitOperator

from datetime import datetime
import requests


def ingest_data():
    import pandas as pd
    import os

    def _first_existing_col(frame: pd.DataFrame, *candidates: str):
        for col_name in candidates:
            if col_name in frame.columns:
                return frame[col_name]
        return None

    def _to_boolish_str(series: pd.Series | None):
        if series is None:
            return None
        # Keep original text values but normalize NaNs to None
        return series.astype(object).where(series.notna(), None)

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

    # Also persist a normalized subset to Postgres (raw_toilets) for SQL exploration.
    # This is best-effort: the upstream dataset schema can change.
    try:
        import psycopg2
        from psycopg2.extras import execute_values

        db_host = os.getenv("DB_HOST", "postgres")
        db_port = int(os.getenv("DB_PORT", "5432"))
        db_name = os.getenv("DB_NAME", "airflow")
        db_user = os.getenv("DB_USER", "airflow")
        db_password = os.getenv("DB_PASSWORD", "airflow")

        id_col = _first_existing_col(df, "_id")
        lat_col = _first_existing_col(df, "Latitude")
        lon_col = _first_existing_col(df, "Longitude")
        name_col = _first_existing_col(df, "Name")
        address_col = _first_existing_col(df, "Address", "Address1")
        suburb_col = _first_existing_col(df, "Suburb", "Town")
        postcode_col = _first_existing_col(df, "Postcode")
        group_col = _first_existing_col(df, "Group", "State")
        category_col = _first_existing_col(df, "Category", "FacilityType")
        accessible_col = _first_existing_col(df, "Accessible")
        changing_col = _first_existing_col(df, "ChangingPlace", "ChangingPlaces")
        gender_col = _first_existing_col(df, "Gender")

        if id_col is None:
            raise RuntimeError("Missing _id column in API response; cannot load raw_toilets")

        raw_df = pd.DataFrame(
            {
                "_id": id_col.astype(str),
                "Latitude": lat_col,
                "Longitude": lon_col,
                "Name": name_col,
                "Address": address_col,
                "Suburb": suburb_col,
                "Postcode": postcode_col,
                "Group": group_col,
                "Category": category_col,
                "Accessible": accessible_col,
                "ChangingPlace": changing_col,
                "Gender": gender_col,
            }
        )

        # Replace NaNs with None for psycopg2
        raw_df = raw_df.astype(object).where(raw_df.notna(), None)

        rows = [tuple(r) for r in raw_df.itertuples(index=False, name=None)]
        if rows:
            conn = psycopg2.connect(
                host=db_host,
                port=db_port,
                dbname=db_name,
                user=db_user,
                password=db_password,
            )
            try:
                with conn.cursor() as cur:
                    sql = """
                    INSERT INTO raw_toilets
                    (_id, Latitude, Longitude, Name, Address, Suburb, Postcode, "Group", Category, Accessible, ChangingPlace, Gender)
                    VALUES %s
                    ON CONFLICT (_id) DO UPDATE SET
                        Latitude = EXCLUDED.Latitude,
                        Longitude = EXCLUDED.Longitude,
                        Name = EXCLUDED.Name,
                        Address = EXCLUDED.Address,
                        Suburb = EXCLUDED.Suburb,
                        Postcode = EXCLUDED.Postcode,
                        "Group" = EXCLUDED."Group",
                        Category = EXCLUDED.Category,
                        Accessible = EXCLUDED.Accessible,
                        ChangingPlace = EXCLUDED.ChangingPlace,
                        Gender = EXCLUDED.Gender,
                        loaded_at = CURRENT_TIMESTAMP
                    """
                    execute_values(cur, sql, rows, page_size=1000)
                conn.commit()
            finally:
                conn.close()
    except Exception as e:
        # Don't fail the DAG for raw DB load issues; CSV remains the main artifact.
        print(f"WARN: could not load raw_toilets to Postgres: {e}")

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