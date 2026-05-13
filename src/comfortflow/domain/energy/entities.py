from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class EnergyForecast:
    kwh_next_6h: float
    model_name: str


@dataclass(frozen=True)
class EnergyReading:
    timestamp_utc: str
    building_id: str
    zone_id: str
    consumption_kwh: float
