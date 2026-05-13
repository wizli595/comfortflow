"""Spark Job: silver -> gold (feature engineering)."""

from __future__ import annotations

import math

from pyspark.sql import DataFrame
from pyspark.sql import functions as F
from pyspark.sql.types import DoubleType

from comfortflow.spark_jobs.spark_session import create_spark_session

SILVER_PATH = "s3a://silver/ashrae/cleaned.parquet"
GOLD_PATH = "s3a://gold/comfort/features.parquet"


def main() -> None:
    spark = create_spark_session("silver_to_gold")
    silver = spark.read.parquet(SILVER_PATH)
    gold = run(silver)
    gold.write.mode("overwrite").parquet(GOLD_PATH)
    print(f"Gold: {gold.count()} rows, {len(gold.columns)} columns written to {GOLD_PATH}")
    spark.stop()


def run(df: DataFrame) -> DataFrame:
    df = add_ppd_estimate(df)
    df = encode_season(df)
    df = encode_building_type(df)
    df = add_temp_humidity_interaction(df)
    return df


def add_ppd_estimate(df: DataFrame) -> DataFrame:
    """PPD = 100 - 95 * exp(-0.03353 * PMV^4 - 0.2179 * PMV^2), using thermal_sensation as PMV proxy."""
    return df.withColumn(
        "ppd_estimate",
        F.lit(100.0) - F.lit(95.0) * F.exp(
            F.lit(-0.03353) * F.pow(F.col("thermal_sensation"), 4)
            - F.lit(0.2179) * F.pow(F.col("thermal_sensation"), 2)
        ),
    )


def encode_season(df: DataFrame) -> DataFrame:
    return (
        df.withColumn("is_summer", (F.col("season") == "Summer").cast("int"))
        .withColumn("is_winter", (F.col("season") == "Winter").cast("int"))
    )


def encode_building_type(df: DataFrame) -> DataFrame:
    return (
        df.withColumn("is_office", (F.col("building_type") == "Office").cast("int"))
        .withColumn("is_classroom", (F.col("building_type") == "Classroom").cast("int"))
        .withColumn("is_residential", (F.col("building_type") == "Multifamily housing").cast("int"))
    )


def add_temp_humidity_interaction(df: DataFrame) -> DataFrame:
    return df.withColumn(
        "temp_humidity_index",
        F.col("air_temperature_celsius") * F.col("relative_humidity_percent") / 100.0,
    )


if __name__ == "__main__":
    main()
