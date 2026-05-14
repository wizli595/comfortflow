"""Feature repository backed by local Parquet files."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from sklearn.model_selection import train_test_split

FEATURE_COLS = [
    "air_temperature_celsius",
    "relative_humidity_percent",
    "air_velocity_ms",
    "clothing_insulation_clo",
    "metabolic_rate_met",
]
TARGET_COL = "thermal_sensation"


class ParquetFeatureRepository:
    def __init__(self, gold_path: str = "data/gold/comfort/features.parquet") -> None:
        df = pq.read_table(gold_path).to_pandas()
        X = df[FEATURE_COLS].values
        y = df[TARGET_COL].values
        self._X_train, self._X_test, self._y_train, self._y_test = train_test_split(
            X, y, test_size=0.2, random_state=42,
        )

    def get_comfort_features(self, building_id: str = "all") -> np.ndarray:
        return self._X_train

    def get_comfort_targets(self, building_id: str = "all") -> np.ndarray:
        return self._y_train

    @property
    def test_features(self) -> np.ndarray:
        return self._X_test

    @property
    def test_targets(self) -> np.ndarray:
        return self._y_test
