"""Spark Job: bronze -> silver (clean and normalize)."""

from __future__ import annotations

from pyspark.sql import DataFrame
from pyspark.sql import functions as F
from pyspark.sql.types import DoubleType

from comfortflow.spark_jobs.spark_session import create_spark_session

BRONZE_PATH = "s3a://bronze/ashrae/ashrae_db2.parquet"
SILVER_PATH = "s3a://silver/ashrae/cleaned.parquet"

# Column index -> clean name
COLUMN_RENAMES = {
    14: "thermal_sensation",
    29: "air_temperature_celsius",
    49: "relative_humidity_percent",
    52: "air_velocity_ms",
    23: "clothing_insulation_clo",
    24: "metabolic_rate_met",
    5: "climate",
    8: "building_type",
    3: "season",
    7: "country",
}

NUMERIC_COLS = [
    "thermal_sensation",
    "air_temperature_celsius",
    "relative_humidity_percent",
    "air_velocity_ms",
    "clothing_insulation_clo",
    "metabolic_rate_met",
]

OUTLIER_BOUNDS = {
    "air_temperature_celsius": (5.0, 45.0),
    "relative_humidity_percent": (0.0, 100.0),
    "air_velocity_ms": (0.0, 5.0),
    "clothing_insulation_clo": (0.0, 3.0),
    "metabolic_rate_met": (0.5, 4.0),
    "thermal_sensation": (-3.0, 3.0),
}


def main() -> None:
    spark = create_spark_session("bronze_to_silver")
    raw = spark.read.parquet(BRONZE_PATH)
    cleaned = run(raw)
    cleaned.write.mode("overwrite").parquet(SILVER_PATH)
    print(f"Silver: {cleaned.count()} rows written to {SILVER_PATH}")
    spark.stop()


def run(raw: DataFrame) -> DataFrame:
    df = select_and_rename(raw)
    df = cast_numeric(df)
    df = drop_nulls(df)
    df = remove_outliers(df)
    return df


def select_and_rename(raw: DataFrame) -> DataFrame:
    cols = raw.columns
    selected = [F.col(f"`{cols[idx]}`").alias(name) for idx, name in COLUMN_RENAMES.items()]
    return raw.select(selected)


def cast_numeric(df: DataFrame) -> DataFrame:
    for col in NUMERIC_COLS:
        df = df.withColumn(col, F.col(col).cast(DoubleType()))
    return df


def drop_nulls(df: DataFrame) -> DataFrame:
    return df.dropna(subset=NUMERIC_COLS)


def remove_outliers(df: DataFrame) -> DataFrame:
    for col, (lo, hi) in OUTLIER_BOUNDS.items():
        df = df.filter(F.col(col).between(lo, hi))
    return df


if __name__ == "__main__":
    main()
