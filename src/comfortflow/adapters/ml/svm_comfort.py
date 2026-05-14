"""SVM implementation of ComfortModel protocol."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR


class SVMComfortModel:
    def __init__(self, max_train_samples: int = 20_000) -> None:
        self._model: SVR | None = None
        self._scaler = StandardScaler()
        self._metrics: dict[str, float] = {}
        self._max_train_samples = max_train_samples

    def train(self, features: np.ndarray, target: np.ndarray) -> None:
        X_tr, X_val, y_tr, y_val = train_test_split(features, target, test_size=0.1, random_state=42)
        if len(X_tr) > self._max_train_samples:
            idx = np.random.RandomState(42).choice(len(X_tr), self._max_train_samples, replace=False)
            X_tr, y_tr = X_tr[idx], y_tr[idx]
        X_tr = self._scaler.fit_transform(X_tr)
        X_val = self._scaler.transform(X_val)
        self._model = SVR(kernel="rbf", C=1.0, gamma="scale")
        self._model.fit(X_tr, y_tr)
        preds = self._model.predict(X_val)
        self._metrics = {
            "rmse": float(np.sqrt(mean_squared_error(y_val, preds))),
            "mae": float(mean_absolute_error(y_val, preds)),
            "r2": float(r2_score(y_val, preds)),
        }

    def predict(self, features: np.ndarray) -> np.ndarray:
        return self._model.predict(self._scaler.transform(features))

    def get_metrics(self) -> dict[str, float]:
        return self._metrics
