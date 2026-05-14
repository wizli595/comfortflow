"""Spark Job: silver -> gold (feature engineering)."""

from pyspark.sql import DataFrame
from pyspark.sql import functions as F

SILVER_PATH = "data/silver/cleaned.parquet"
GOLD_PATH = "data/gold/comfort/features.parquet"


def run(df: DataFrame) -> DataFrame:
    df = df.withColumn(
        "ppd_estimate",
        F.lit(100.0) - F.lit(95.0) * F.exp(
            F.lit(-0.03353) * F.pow(F.col("thermal_sensation"), 4)
            - F.lit(0.2179) * F.pow(F.col("thermal_sensation"), 2)
        ),
    )
    df = df.withColumn("is_summer", (F.col("season") == "Summer").cast("int"))
    df = df.withColumn("is_winter", (F.col("season") == "Winter").cast("int"))
    df = df.withColumn("is_office", (F.col("building_type") == "Office").cast("int"))
    df = df.withColumn("is_classroom", (F.col("building_type") == "Classroom").cast("int"))
    df = df.withColumn(
        "temp_humidity_index",
        F.col("air_temperature_celsius") * F.col("relative_humidity_percent") / 100.0,
    )
    return df


if __name__ == "__main__":
    import os
    import pyarrow as pa
    import pyarrow.parquet as pq
    from comfortflow.spark_jobs.spark_session import create_spark_session

    spark = create_spark_session("silver_to_gold")
    silver = spark.read.parquet(SILVER_PATH)
    gold = run(silver)

    pdf = gold.toPandas()
    os.makedirs(os.path.dirname(GOLD_PATH), exist_ok=True)
    pq.write_table(pa.Table.from_pandas(pdf), GOLD_PATH)
    print(f"Gold: {len(pdf):,} rows, {len(pdf.columns)} cols -> {GOLD_PATH}")
    spark.stop()
