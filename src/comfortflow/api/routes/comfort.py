"""Comfort prediction routes."""

from __future__ import annotations

import numpy as np
from fastapi import APIRouter, Depends

from comfortflow.api.dependencies import get_comfort_model
from comfortflow.api.schemas import ComfortRequest, ComfortResponse
from comfortflow.domain.comfort.entities import IndoorClimate, OccupantState
from comfortflow.domain.comfort.pmv import calculate_pmv, calculate_ppd

router = APIRouter(prefix="/comfort", tags=["comfort"])


@router.post("/predict", response_model=ComfortResponse)
def predict_comfort(req: ComfortRequest, model=Depends(get_comfort_model)):
    features = np.array([[
        req.air_temperature_celsius,
        req.relative_humidity_percent,
        req.air_velocity_ms,
        req.clothing_insulation_clo,
        req.metabolic_rate_met,
    ]])
    prediction = float(model.predict(features)[0])

    climate = IndoorClimate(
        req.air_temperature_celsius,
        req.air_temperature_celsius,
        req.relative_humidity_percent,
        req.air_velocity_ms,
    )
    occupant = OccupantState(req.clothing_insulation_clo, req.metabolic_rate_met)
    pmv = calculate_pmv(climate, occupant)
    ppd = calculate_ppd(pmv)

    sensations = {-3: "Cold", -2: "Cool", -1: "Slightly Cool", 0: "Neutral",
                  1: "Slightly Warm", 2: "Warm", 3: "Hot"}
    label = sensations.get(round(prediction), "Neutral")

    return ComfortResponse(
        pmv=pmv.value,
        ppd=ppd.value,
        thermal_sensation=label,
        model_used="XGBoostComfortModel",
    )
