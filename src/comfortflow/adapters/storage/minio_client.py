"""MinIO connection helper."""

from __future__ import annotations

from minio import Minio


def create_minio_client(
    endpoint: str = "localhost:9000",
    access_key: str = "comfortflow",
    secret_key: str = "comfortflow123",
) -> Minio:
    return Minio(endpoint, access_key=access_key, secret_key=secret_key, secure=False)


def ensure_bucket(client: Minio, bucket: str) -> None:
    if not client.bucket_exists(bucket):
        client.make_bucket(bucket)
