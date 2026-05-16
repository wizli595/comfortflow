"""Load Sinergym energy data for training."""

from __future__ import annotations

import numpy as np
import pandas as pd

from comfortflow.config import get_config

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
    def __init__(self, path: str | None = None) -> None:
        path = path or get_config()["paths"]["sinergym_csv"]
        df = pd.read_csv(path)
        rename = {f"obs_{i}": OBS_NAMES[i] for i in range(len(OBS_NAMES))}
        df = df.rename(columns=rename)
        self._features = df[FEATURE_COLS].values
        self._targets = df[TARGET_COL].values / 1000.0

    def get_energy_features(self) -> np.ndarray:
        return self._features

    def get_energy_targets(self) -> np.ndarray:
        return self._targets
