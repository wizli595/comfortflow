"""Spark Job: bronze -> silver (clean and normalize)."""

from pyspark.sql import DataFrame
from pyspark.sql import functions as F
from pyspark.sql.types import DoubleType

BRONZE_PATH = "data/bronze/ashrae.parquet"
SILVER_PATH = "data/silver/cleaned.parquet"

COLUMN_RENAMES = {
    14: "thermal_sensation",
    29: "air_temperature_celsius",
    49: "relative_humidity_percent",
    52: "air_velocity_ms",
    23: "clothing_insulation_clo",
    24: "metabolic_rate_met",
    5: "climate", 8: "building_type", 3: "season", 7: "country",
}

NUMERIC_COLS = [
    "thermal_sensation", "air_temperature_celsius",
    "relative_humidity_percent", "air_velocity_ms",
    "clothing_insulation_clo", "metabolic_rate_met",
]

OUTLIER_BOUNDS = {
    "air_temperature_celsius": (5.0, 45.0),
    "relative_humidity_percent": (0.0, 100.0),
    "air_velocity_ms": (0.0, 5.0),
    "clothing_insulation_clo": (0.0, 3.0),
    "metabolic_rate_met": (0.5, 4.0),
    "thermal_sensation": (-3.0, 3.0),
}


def run(raw: DataFrame) -> DataFrame:
    cols = raw.columns
    selected = [F.col(f"`{cols[idx]}`").alias(name) for idx, name in COLUMN_RENAMES.items()]
    df = raw.select(selected)
    for col in NUMERIC_COLS:
        df = df.withColumn(col, F.col(col).cast(DoubleType()))
    df = df.dropna(subset=NUMERIC_COLS)
    for col, (lo, hi) in OUTLIER_BOUNDS.items():
        df = df.filter(F.col(col).between(lo, hi))
    return df


if __name__ == "__main__":
    import os
    import pyarrow as pa
    import pyarrow.parquet as pq
    from comfortflow.spark_jobs.spark_session import create_spark_session

    spark = create_spark_session("bronze_to_silver")
    raw = spark.read.parquet(BRONZE_PATH)
    cleaned = run(raw)

    pdf = cleaned.toPandas()
    os.makedirs(os.path.dirname(SILVER_PATH), exist_ok=True)
    pq.write_table(pa.Table.from_pandas(pdf), SILVER_PATH)
    print(f"Silver: {len(pdf):,} rows -> {SILVER_PATH}")
    spark.stop()
