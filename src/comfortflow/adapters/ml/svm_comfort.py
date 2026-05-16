"""SVM implementation of ComfortModel protocol."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR


class SVMComfortModel:
    def __init__(self, max_samples: int = 20_000) -> None:
        self._model: SVR | None = None
        self._scaler = StandardScaler()
        self._metrics: dict[str, float] = {}
        self._max_samples = max_samples

    def train(self, features: np.ndarray, target: np.ndarray) -> None:
        if len(features) > self._max_samples:
            idx = np.random.RandomState(42).choice(len(features), self._max_samples, replace=False)
            features, target = features[idx], target[idx]
        X_scaled = self._scaler.fit_transform(features)
        self._model = SVR(kernel="rbf", C=1.0, gamma="scale")
        self._model.fit(X_scaled, target)
        preds = self._model.predict(X_scaled)
        self._metrics = {
            "rmse": float(np.sqrt(mean_squared_error(target, preds))),
            "mae": float(mean_absolute_error(target, preds)),
            "r2": float(r2_score(target, preds)),
        }

    def predict(self, features: np.ndarray) -> np.ndarray:
        return self._model.predict(self._scaler.transform(features))

    def get_metrics(self) -> dict[str, float]:
        return self._metrics
