"""Load Sinergym energy data for training."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

OBS_NAMES = [
    "month", "day_of_month", "hour", "outdoor_temperature", "outdoor_humidity",
    "wind_speed", "wind_direction", "diffuse_solar_radiation", "direct_solar_radiation",
    "air_temperature", "air_humidity", "people_occupant", "heating_setpoint",
    "cooling_setpoint", "co2_emission", "hvac_electricity_demand_rate",
    "total_electricity_hvac",
]

FEATURE_COLS = [
    "outdoor_temperature", "outdoor_humidity", "wind_speed",
    "air_temperature", "air_humidity", "people_occupant",
    "heating_setpoint", "cooling_setpoint", "hour", "month",
]
TARGET_COL = "hvac_electricity_demand_rate"


class EnergyRepository:
    def __init__(self, path: str = "data/sinergym/energy_data.csv") -> None:
        df = pd.read_csv(path)
        rename = {f"obs_{i}": OBS_NAMES[i] for i in range(len(OBS_NAMES))}
        rename["action_0"] = "action_heating_setpoint"
        rename["action_1"] = "action_cooling_setpoint"
        df = df.rename(columns=rename)

        X = df[FEATURE_COLS].values
        y = df[TARGET_COL].values / 1000.0  # Convert to kW

        self._X_train, self._X_test, self._y_train, self._y_test = train_test_split(
            X, y, test_size=0.2, random_state=42,
        )

    def get_energy_features(self) -> np.ndarray:
        return self._X_train

    def get_energy_targets(self) -> np.ndarray:
        return self._y_train

    @property
    def test_features(self) -> np.ndarray:
        return self._X_test

    @property
    def test_targets(self) -> np.ndarray:
        return self._y_test
