from pyspark.sql import SparkSession
from pyspark.sql.functions import floor, mean, stddev, col, when, count, lit, lower, coalesce
from pyspark.sql.types import DoubleType
import os

spark = SparkSession.builder \
    .appName("Toilet_LR1") \
    .config("spark.hadoop.fs.permissions.umask-mode", "000") \
    .master("spark://spark-master:7077") \
    .getOrCreate()

spark.sparkContext.setLogLevel("WARN")


DATA_DIR = "/opt/data"
RAW_FILE = f"{DATA_DIR}/toilets.csv"
PROCESSED_DIR = f"{DATA_DIR}/processed"

DB_HOST = os.getenv("DB_HOST", "postgres")
DB_PORT = os.getenv("DB_PORT", "5432")
DB_NAME = os.getenv("DB_NAME", "airflow")
DB_USER = os.getenv("DB_USER", "airflow")
DB_PASSWORD = os.getenv("DB_PASSWORD", "airflow")

JDBC_URL = f"jdbc:postgresql://{DB_HOST}:{DB_PORT}/{DB_NAME}"
JDBC_PROPERTIES = {
    "user": DB_USER,
    "password": DB_PASSWORD,
    "driver": "org.postgresql.Driver"
}


GRID_SIZE = 0.01


def col_or_null(frame, name: str):
    return col(name) if name in frame.columns else lit(None)

def is_true(c):
    return lower(c.cast("string")).isin(["true", "yes", "1"])

try:
    raw = spark.read.csv(RAW_FILE, header=True, inferSchema=False)

    gender_from_flags = (
        when(is_true(col_or_null(raw, "Unisex")), lit("Unisex"))
        .when(is_true(col_or_null(raw, "AllGender")), lit("AllGender"))
        .when(is_true(col_or_null(raw, "Male")) & is_true(col_or_null(raw, "Female")), lit("Male/Female"))
        .when(is_true(col_or_null(raw, "Male")), lit("Male"))
        .when(is_true(col_or_null(raw, "Female")), lit("Female"))
        .otherwise(lit(None))
    )

    df = raw.select(
        col_or_null(raw, "_id").cast("string").alias("_id"),
        col_or_null(raw, "Latitude").cast(DoubleType()).alias("Latitude"),
        col_or_null(raw, "Longitude").cast(DoubleType()).alias("Longitude"),
        col_or_null(raw, "Name").cast("string").alias("Name"),
        coalesce(col_or_null(raw, "Address"), col_or_null(raw, "Address1")).cast("string").alias("Address"),
        coalesce(col_or_null(raw, "Suburb"), col_or_null(raw, "Town")).cast("string").alias("Suburb"),
        col_or_null(raw, "Postcode").cast("string").alias("Postcode"),
        coalesce(col_or_null(raw, "Group"), col_or_null(raw, "State")).cast("string").alias("Group"),
        coalesce(col_or_null(raw, "Category"), col_or_null(raw, "FacilityType")).cast("string").alias("Category"),
        col_or_null(raw, "Accessible").cast("string").alias("Accessible"),
        coalesce(col_or_null(raw, "ChangingPlace"), col_or_null(raw, "ChangingPlaces")).cast("string").alias(
            "ChangingPlace"
        ),
        coalesce(col_or_null(raw, "Gender"), gender_from_flags).cast("string").alias("Gender"),
    )

    print("Loaded data from local file")
except Exception as e:
    print(f"Error loading data: {e}")
    raise

print(f"Total records loaded: {df.count()}")


print("Cleaning data...")

df = df.dropna(subset=["Latitude", "Longitude"])

df = df.dropDuplicates()

df = df.filter(
    (col("Latitude").between(-90, 90)) & 
    (col("Longitude").between(-180, 180))
)

print(f"Records after cleaning: {df.count()}")




df = df.withColumn("lat_bin", floor(col("Latitude") / GRID_SIZE)) \
       .withColumn("lon_bin", floor(col("Longitude") / GRID_SIZE))

df = df.withColumn("is_accessible", 
    when(col("Accessible").isin(["Yes", "yes", "YES", "True", "true"]), 1).otherwise(0))

df = df.withColumn("gender_type",
    when(col("Gender").contains("Unisex"), "unisex")
    .when(col("Gender").contains("Male"), "male")
    .when(col("Gender").contains("Female"), "female")
    .otherwise("unknown")
)


print("Computing aggregations...")

agg = df.groupBy("lat_bin", "lon_bin").count() \
        .withColumnRenamed("count", "toilet_count")

stats = agg.select(
    mean("toilet_count").alias("mean"),
    stddev("toilet_count").alias("std")
).collect()[0]

mean_val = stats["mean"] if stats["mean"] is not None else 0
std_val = stats["std"] if stats["std"] is not None else 1

agg = agg.withColumn(
    "z_score",
    (col("toilet_count") - mean_val) / std_val
)

anomalies = agg.filter(col("z_score") < -2)

hotspots = agg.filter(col("z_score") > 2)



suburb_stats = df.groupBy("Suburb", "Postcode").agg(
    mean("Latitude").alias("avg_lat"),
    mean("Longitude").alias("avg_lon"),
    mean("is_accessible").alias("accessibility_rate"),
    count("*").alias("toilet_count")
).orderBy(col("toilet_count").desc())


print("Saving results to Parquet...")

agg.write.mode("overwrite").parquet(f"{PROCESSED_DIR}/grid_stats")
anomalies.write.mode("overwrite").parquet(f"{PROCESSED_DIR}/anomalies")
hotspots.write.mode("overwrite").parquet(f"{PROCESSED_DIR}/hotspots")
suburb_stats.write.mode("overwrite").parquet(f"{PROCESSED_DIR}/suburb_stats")

print(f"Results saved to {PROCESSED_DIR}")


print("Uploading results to Postgres...")

try:
    agg.write.jdbc(
        url=JDBC_URL,
        table="toilets_grid_stats",
        mode="overwrite",
        properties=JDBC_PROPERTIES
    )
    print("Uploaded grid_stats to Postgres")

    anomalies.write.jdbc(
        url=JDBC_URL,
        table="toilets_anomalies",
        mode="overwrite",
        properties=JDBC_PROPERTIES
    )

    hotspots.write.jdbc(
        url=JDBC_URL,
        table="toilets_hotspots",
        mode="overwrite",
        properties=JDBC_PROPERTIES
    )

    suburb_stats.write.jdbc(
        url=JDBC_URL,
        table="toilets_suburb_stats",
        mode="overwrite",
        properties=JDBC_PROPERTIES
    )
    print("Uploaded suburb_stats to Postgres")

except Exception as e:
    print(f"Error uploading to Postgres: {e}")
    raise



print(f"Total records processed: {df.count()}")
print(f"Grid cells created: {agg.count()}")
print(f"Anomalies detected (low density): {anomalies.count()}")
print(f"Hotspots detected (high density): {hotspots.count()}")
print(f"Suburbs analyzed: {suburb_stats.count()}")

spark.stop()
