"""Create a local Spark session."""

from __future__ import annotations

import os

from pyspark.sql import SparkSession


def create_spark_session(app_name: str = "comfortflow") -> SparkSession:
    os.environ.setdefault("HADOOP_HOME", "C:/hadoop")
    return (
        SparkSession.builder
        .appName(app_name)
        .master("local[*]")
        .getOrCreate()
    )
