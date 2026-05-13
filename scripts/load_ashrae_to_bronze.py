"""Load ASHRAE CSV into MinIO bronze/ as Parquet."""

from pathlib import Path
import io

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from comfortflow.adapters.storage.minio_client import create_minio_client, ensure_bucket

BUCKET = "bronze"
OBJECT_KEY = "ashrae/ashrae_db2.parquet"
CSV_PATH = Path("data/ashrae_db2.01.csv")


def main() -> None:
    print("Reading CSV...")
    df = pd.read_csv(CSV_PATH, encoding="latin-1", low_memory=False)
    print(f"  {len(df):,} rows, {len(df.columns)} columns")

    print("Converting to Parquet...")
    table = pa.Table.from_pandas(df)
    buf = io.BytesIO()
    pq.write_table(table, buf)
    buf.seek(0)
    size = buf.getbuffer().nbytes

    print("Uploading to MinIO bronze/ashrae/...")
    client = create_minio_client()
    ensure_bucket(client, BUCKET)
    client.put_object(BUCKET, OBJECT_KEY, buf, length=size, content_type="application/octet-stream")

    print(f"Done. Uploaded {size / 1024 / 1024:.1f} MB to {BUCKET}/{OBJECT_KEY}")


if __name__ == "__main__":
    main()
