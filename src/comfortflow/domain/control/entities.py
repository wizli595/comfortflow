from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class HVACSetpoints:
    heating_celsius: float
    cooling_celsius: float

    def __post_init__(self) -> None:
        if self.heating_celsius >= self.cooling_celsius:
            raise ValueError("Heating setpoint must be below cooling setpoint")


@dataclass(frozen=True)
class BuildingState:
    indoor_temperature_celsius: float
    outdoor_temperature_celsius: float
    indoor_humidity_percent: float
    current_heating_setpoint: float
    current_cooling_setpoint: float
    hour_of_day: int
    pmv_prediction: float
