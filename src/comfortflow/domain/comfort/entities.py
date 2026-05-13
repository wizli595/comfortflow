from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum


class ThermalSensation(IntEnum):
    COLD = -3
    COOL = -2
    SLIGHTLY_COOL = -1
    NEUTRAL = 0
    SLIGHTLY_WARM = 1
    WARM = 2
    HOT = 3


@dataclass(frozen=True)
class PMVScore:
    value: float

    def __post_init__(self) -> None:
        if not -3.0 <= self.value <= 3.0:
            raise ValueError(f"PMV must be in [-3, +3], got {self.value}")

    @property
    def is_comfortable(self) -> bool:
        return abs(self.value) <= 0.5


@dataclass(frozen=True)
class PPDScore:
    value: float

    def __post_init__(self) -> None:
        if not 0.0 <= self.value <= 100.0:
            raise ValueError(f"PPD must be in [0, 100], got {self.value}")


@dataclass(frozen=True)
class IndoorClimate:
    air_temperature_celsius: float
    radiant_temperature_celsius: float
    relative_humidity_percent: float
    air_velocity_ms: float


@dataclass(frozen=True)
class OccupantState:
    clothing_insulation_clo: float
    metabolic_rate_met: float
