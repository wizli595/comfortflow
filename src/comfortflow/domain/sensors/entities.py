from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SensorReading:
    timestamp_utc: str
    building_id: str
    zone_id: str
    air_temperature_celsius: float
    relative_humidity_percent: float
    co2_ppm: float | None = None
    air_velocity_ms: float | None = None
