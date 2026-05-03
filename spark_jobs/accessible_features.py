"""spark_jobs/accessible_features.py

Feature engineering для задачи бинарной классификации:
предсказать доступность туалета (Accessible: True/False) по признакам.

1 строка = 1 туалет (НЕ агрегация по grid), чтобы датасет был достаточного размера.

Вход:
- /opt/data/toilets.csv (Lab1 ingest_api_data)

Выход:
- Parquet: /opt/data/processed/ml_accessible_features
- Postgres: table ml_accessible_features

Запуск:
  spark-submit /opt/spark_jobs/accessible_features.py
"""

from __future__ import annotations

import os

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import DoubleType


def main() -> None:
    spark = (
        SparkSession.builder.appName("Toilet_Accessible_Features")
        .config("spark.hadoop.fs.permissions.umask-mode", "000")
        .master("spark://spark-master:7077")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("WARN")

    data_dir = "/opt/data"
    raw_file = f"{data_dir}/toilets.csv"
    processed_dir = f"{data_dir}/processed"

    db_host = os.getenv("DB_HOST", "postgres")
    db_port = os.getenv("DB_PORT", "5432")
    db_name = os.getenv("DB_NAME", "airflow")
    db_user = os.getenv("DB_USER", "airflow")
    db_password = os.getenv("DB_PASSWORD", "airflow")

    jdbc_url = f"jdbc:postgresql://{db_host}:{db_port}/{db_name}"

    grid_size = 0.01  # keep consistent with other jobs

    raw = spark.read.csv(raw_file, header=True, inferSchema=False)

    def col_or_null(name: str):
        return F.col(name) if name in raw.columns else F.lit(None)

    def is_true(c):
        return F.lower(c.cast("string")).isin(["true", "yes", "1"])

    gender_from_flags = (
        F.when(is_true(col_or_null("Unisex")), F.lit("Unisex"))
        .when(is_true(col_or_null("AllGender")), F.lit("AllGender"))
        .when(is_true(col_or_null("Male")) & is_true(col_or_null("Female")), F.lit("Male/Female"))
        .when(is_true(col_or_null("Male")), F.lit("Male"))
        .when(is_true(col_or_null("Female")), F.lit("Female"))
        .otherwise(F.lit(None))
    )

    df = raw.select(
        col_or_null("_id").cast("string").alias("_id"),
        col_or_null("Latitude").cast(DoubleType()).alias("Latitude"),
        col_or_null("Longitude").cast(DoubleType()).alias("Longitude"),
        col_or_null("Name").cast("string").alias("Name"),
        F.coalesce(col_or_null("Address"), col_or_null("Address1")).cast("string").alias("Address"),
        F.coalesce(col_or_null("Suburb"), col_or_null("Town")).cast("string").alias("Suburb"),
        col_or_null("Postcode").cast("string").alias("Postcode"),
        F.coalesce(col_or_null("Group"), col_or_null("State")).cast("string").alias("Group"),
        F.coalesce(col_or_null("Category"), col_or_null("FacilityType")).cast("string").alias("Category"),
        col_or_null("Accessible").cast("string").alias("Accessible"),
        F.coalesce(col_or_null("ChangingPlace"), col_or_null("ChangingPlaces")).cast("string").alias(
            "ChangingPlace"
        ),
        F.coalesce(col_or_null("Gender"), gender_from_flags).cast("string").alias("Gender"),
    )

    df = df.dropna(subset=["_id", "Latitude", "Longitude"]).dropDuplicates(["_id"])

    # Target
    df = df.withColumn(
        "is_accessible",
        F.when(
            F.col("Accessible").isin(["Yes", "yes", "YES", "TRUE", "true", "True"]),
            F.lit(1),
        ).otherwise(F.lit(0)),
    )

    # Simple derived features
    df = df.withColumn("lat_bin", F.floor(F.col("Latitude") / F.lit(grid_size)).cast("bigint"))
    df = df.withColumn("lon_bin", F.floor(F.col("Longitude") / F.lit(grid_size)).cast("bigint"))

    center = df.agg(F.avg("Latitude").alias("center_lat"), F.avg("Longitude").alias("center_lon")).first()
    center_lat = center["center_lat"] if center and center["center_lat"] is not None else -31.95
    center_lon = center["center_lon"] if center and center["center_lon"] is not None else 115.85

    df = df.withColumn(
        "distance_to_center",
        F.sqrt(
            F.pow(F.col("Latitude") - F.lit(center_lat), 2)
            + F.pow(F.col("Longitude") - F.lit(center_lon), 2)
        )
        * F.lit(111_000.0),
    )

    df = df.withColumn(
        "has_changing_place",
        F.when(
            F.col("ChangingPlace").isin(["Yes", "yes", "YES", "TRUE", "true", "True"]),
            F.lit(1),
        ).otherwise(F.lit(0)),
    )

    df = df.withColumn(
        "gender_encoded",
        F.when(F.col("Gender").contains("Unisex"), F.lit(0))
        .when(F.col("Gender").contains("Male"), F.lit(1))
        .when(F.col("Gender").contains("Female"), F.lit(2))
        .otherwise(F.lit(3)),
    )

    df = df.withColumn(
        "category_encoded",
        F.when(F.col("Category").contains("Park"), F.lit(1))
        .when(F.col("Category").contains("Sport"), F.lit(2))
        .when(F.col("Category").contains("Shopping"), F.lit(3))
        .otherwise(F.lit(0)),
    )

    # Non-leaky context features
    suburb_counts = df.groupBy("Suburb").agg(F.count(F.lit(1)).cast("bigint").alias("suburb_toilet_count"))
    grid_counts = df.groupBy("lat_bin", "lon_bin").agg(F.count(F.lit(1)).cast("bigint").alias("grid_toilet_count"))

    out = (
        df.join(suburb_counts, on=["Suburb"], how="left")
        .join(grid_counts, on=["lat_bin", "lon_bin"], how="left")
        .select(
            F.col("_id").cast("string").alias("_id"),
            F.col("lat_bin"),
            F.col("lon_bin"),
            F.col("distance_to_center"),
            F.col("gender_encoded"),
            F.col("category_encoded"),
            F.col("has_changing_place"),
            F.col("suburb_toilet_count"),
            F.col("grid_toilet_count"),
            F.col("is_accessible"),
            F.current_timestamp().alias("created_at"),
        )
    )

    out_parquet = f"{processed_dir}/ml_accessible_features"
    out.write.mode("overwrite").parquet(out_parquet)

    (
        out.write.format("jdbc")
        .option("url", jdbc_url)
        .option("dbtable", "ml_accessible_features")
        .option("user", db_user)
        .option("password", db_password)
        .option("driver", "org.postgresql.Driver")
        .option("truncate", "true")
        .mode("overwrite")
        .save()
    )

    print(f"Accessible ML features saved to: {out_parquet}")
    print("Accessible ML features written to Postgres table: ml_accessible_features")
    print(f"Records: {out.count()}")

    spark.stop()


if __name__ == "__main__":
    main()
