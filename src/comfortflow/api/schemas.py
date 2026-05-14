"""API request/response schemas."""

from __future__ import annotations

from pydantic import BaseModel, Field


class ComfortRequest(BaseModel):
    air_temperature_celsius: float = Field(..., ge=5, le=45)
    relative_humidity_percent: float = Field(..., ge=0, le=100)
    air_velocity_ms: float = Field(..., ge=0, le=5)
    clothing_insulation_clo: float = Field(..., ge=0, le=3)
    metabolic_rate_met: float = Field(..., ge=0.5, le=4)


class ComfortResponse(BaseModel):
    pmv: float
    ppd: float
    thermal_sensation: str
    model_used: str
