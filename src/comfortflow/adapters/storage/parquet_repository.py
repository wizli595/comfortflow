"""Feature repository backed by local Parquet files."""

from __future__ import annotations

import numpy as np
import pyarrow.parquet as pq

from comfortflow.config import get_config

FEATURE_COLS = [
    "air_temperature_celsius",
    "relative_humidity_percent",
    "air_velocity_ms",
    "clothing_insulation_clo",
    "metabolic_rate_met",
]
TARGET_COL = "thermal_sensation"


class ParquetFeatureRepository:
    def __init__(self, path: str | None = None) -> None:
        path = path or get_config()["paths"]["gold_comfort"]
        df = pq.read_table(path).to_pandas()
        self._features = df[FEATURE_COLS].values
        self._targets = df[TARGET_COL].values

    def get_comfort_features(self, building_id: str = "all") -> np.ndarray:
        return self._features

    def get_comfort_targets(self, building_id: str = "all") -> np.ndarray:
        return self._targets
