"""Prometheus metrics for ComfortFlow API."""

from __future__ import annotations

from prometheus_client import Counter, Histogram, Gauge

PREDICTION_COUNT = Counter("comfort_predictions_total", "Total comfort predictions served")
PREDICTION_LATENCY = Histogram("comfort_prediction_seconds", "Prediction latency")
CURRENT_PMV = Gauge("comfort_pmv_latest", "Latest PMV prediction")
