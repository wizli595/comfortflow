"""Sync local data/ parquet files to MinIO buckets."""

from __future__ import annotations

from pathlib import Path

from comfortflow.adapters.storage.minio_client import create_minio_client, ensure_bucket


def main() -> None:
    client = create_minio_client()
    sync("data/silver", "silver", client)
    sync("data/gold", "gold", client)
    print("Sync complete.")


def sync(local_dir: str, bucket: str, client) -> None:
    ensure_bucket(client, bucket)
    for path in Path(local_dir).rglob("*.parquet"):
        key = str(path.relative_to(local_dir)).replace("\\", "/")
        client.fput_object(bucket, key, str(path))
        print(f"  {bucket}/{key} ({path.stat().st_size / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
