"""Select the best comfort model by lowest RMSE."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import mean_squared_error

from comfortflow.domain.comfort.protocols import ComfortModel


class ModelSelector:
    def __init__(self) -> None:
        self._models: dict[str, ComfortModel] = {}
        self._scores: dict[str, float] = {}

    def register(self, name: str, model: ComfortModel) -> None:
        self._models[name] = model

    def evaluate(self, features: np.ndarray, targets: np.ndarray) -> None:
        for name, model in self._models.items():
            preds = model.predict(features)
            valid = ~np.isnan(preds)
            rmse = float(np.sqrt(mean_squared_error(targets[valid], preds[valid])))
            self._scores[name] = rmse

    def best_model_name(self) -> str:
        return min(self._scores, key=self._scores.get)

    def best_model(self) -> ComfortModel:
        return self._models[self.best_model_name()]

    def rankings(self) -> list[tuple[str, float]]:
        return sorted(self._scores.items(), key=lambda x: x[1])
