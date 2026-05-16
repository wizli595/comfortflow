"""Random Forest implementation of ComfortModel protocol."""

from __future__ import annotations

import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


class RandomForestComfortModel:
    def __init__(self) -> None:
        self._model: RandomForestRegressor | None = None
        self._metrics: dict[str, float] = {}

    def train(self, features: np.ndarray, target: np.ndarray) -> None:
        self._model = RandomForestRegressor(
            n_estimators=200, min_samples_leaf=5, n_jobs=-1, random_state=42,
        )
        self._model.fit(features, target)
        preds = self._model.predict(features)
        self._metrics = {
            "rmse": float(np.sqrt(mean_squared_error(target, preds))),
            "mae": float(mean_absolute_error(target, preds)),
            "r2": float(r2_score(target, preds)),
        }

    def predict(self, features: np.ndarray) -> np.ndarray:
        return self._model.predict(features)

    def get_metrics(self) -> dict[str, float]:
        return self._metrics
