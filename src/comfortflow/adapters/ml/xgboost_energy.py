"""XGBoost energy predictor implementing EnergyModel protocol."""

from __future__ import annotations

import numpy as np
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


class XGBoostEnergyModel:
    def __init__(self) -> None:
        self._model: xgb.XGBRegressor | None = None
        self._metrics: dict[str, float] = {}

    def train(self, features: np.ndarray, target: np.ndarray) -> None:
        self._model = xgb.XGBRegressor(
            n_estimators=300, max_depth=6, learning_rate=0.1,
            subsample=0.8, colsample_bytree=0.8, random_state=42,
        )
        self._model.fit(features, target, verbose=False)
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
