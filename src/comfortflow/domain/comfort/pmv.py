"""PMV/PPD calculation — ISO 7730 via pythermalcomfort."""

from __future__ import annotations

import math

from pythermalcomfort.models.pmv_ppd_iso import pmv_ppd_iso

from comfortflow.domain.comfort.entities import (
    IndoorClimate,
    OccupantState,
    PMVScore,
    PPDScore,
)


def calculate_pmv(climate: IndoorClimate, occupant: OccupantState) -> PMVScore:
    result = pmv_ppd_iso(
        tdb=climate.air_temperature_celsius,
        tr=climate.radiant_temperature_celsius,
        vr=climate.air_velocity_ms,
        rh=climate.relative_humidity_percent,
        met=occupant.metabolic_rate_met,
        clo=occupant.clothing_insulation_clo,
    )
    pmv_value = float(result.pmv)
    if math.isnan(pmv_value):
        pmv_value = 3.0 if climate.air_temperature_celsius > 26 else -3.0
    return PMVScore(value=round(max(-3.0, min(3.0, pmv_value)), 2))


def calculate_ppd(pmv: PMVScore) -> PPDScore:
    v = pmv.value
    ppd = 100.0 - 95.0 * math.exp(-0.03353 * v**4 - 0.2179 * v**2)
    return PPDScore(value=round(ppd, 2))
