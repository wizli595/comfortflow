"""Run the full bronze -> silver -> gold pipeline using pandas + MinIO."""

from __future__ import annotations

import io
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from comfortflow.adapters.storage.ashrae_loader import (
    COLUMN_INDEX_MAP,
    FEATURE_COLS,
    NUMERIC_COLS,
    load_ashrae,
)
from comfortflow.adapters.storage.minio_client import create_minio_client, ensure_bucket


def main() -> None:
    client = create_minio_client()

    # Bronze -> Silver (clean)
    print("=== Bronze -> Silver ===")
    silver = load_ashrae(Path("data/ashrae_db2.01.csv"))
    print(f"  {len(silver):,} clean rows")
    upload_parquet(client, silver, "silver", "ashrae/cleaned.parquet")

    # Silver -> Gold (features)
    print("=== Silver -> Gold ===")
    gold = engineer_features(silver)
    print(f"  {len(gold):,} rows, {len(gold.columns)} columns")
    upload_parquet(client, gold, "gold", "comfort/features.parquet")

    print("Done. Pipeline complete.")


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["ppd_estimate"] = ppd_from_sensation(df["thermal_sensation"])
    df["is_summer"] = (df["season"] == "Summer").astype(int)
    df["is_winter"] = (df["season"] == "Winter").astype(int)
    df["is_office"] = (df["building_type"] == "Office").astype(int)
    df["is_classroom"] = (df["building_type"] == "Classroom").astype(int)
    df["is_residential"] = (df["building_type"] == "Multifamily housing").astype(int)
    df["temp_humidity_index"] = (
        df["air_temperature_celsius"] * df["relative_humidity_percent"] / 100.0
    )
    return df


def ppd_from_sensation(ts: pd.Series) -> pd.Series:
    import numpy as np
    return 100.0 - 95.0 * np.exp(-0.03353 * ts**4 - 0.2179 * ts**2)


def upload_parquet(client, df: pd.DataFrame, bucket: str, key: str) -> None:
    ensure_bucket(client, bucket)
    table = pa.Table.from_pandas(df)
    buf = io.BytesIO()
    pq.write_table(table, buf)
    buf.seek(0)
    size = buf.getbuffer().nbytes
    client.put_object(bucket, key, buf, length=size, content_type="application/octet-stream")
    print(f"  Uploaded {size / 1024 / 1024:.1f} MB to {bucket}/{key}")


if __name__ == "__main__":
    main()
