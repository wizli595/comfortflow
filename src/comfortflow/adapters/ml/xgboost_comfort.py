"""XGBoost implementation of ComfortModel protocol."""

from __future__ import annotations

import numpy as np
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split


class XGBoostComfortModel:
    def __init__(self) -> None:
        self._model: xgb.XGBRegressor | None = None
        self._metrics: dict[str, float] = {}

    def train(self, features: np.ndarray, target: np.ndarray) -> None:
        X_train, X_val, y_train, y_val = train_test_split(
            features, target, test_size=0.1, random_state=42,
        )
        self._model = xgb.XGBRegressor(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
        )
        self._model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=False,
        )
        predictions = self._model.predict(X_val)
        self._metrics = {
            "rmse": float(np.sqrt(mean_squared_error(y_val, predictions))),
            "mae": float(mean_absolute_error(y_val, predictions)),
            "r2": float(r2_score(y_val, predictions)),
        }

    def predict(self, features: np.ndarray) -> np.ndarray:
        return self._model.predict(features)

    def get_metrics(self) -> dict[str, float]:
        return self._metrics
