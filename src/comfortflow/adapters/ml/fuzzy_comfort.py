"""Fuzzy Logic implementation of ComfortModel protocol."""

from __future__ import annotations

import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


class FuzzyComfortModel:
    def __init__(self) -> None:
        self._system: ctrl.ControlSystemSimulation | None = None
        self._metrics: dict[str, float] = {}
        self._build_system()

    def _build_system(self) -> None:
        temp = ctrl.Antecedent(np.arange(5, 46, 0.5), "temperature")
        humidity = ctrl.Antecedent(np.arange(0, 101, 1), "humidity")
        comfort = ctrl.Consequent(np.arange(-3, 3.1, 0.1), "comfort")

        temp["cold"] = fuzz.trimf(temp.universe, [5, 5, 18])
        temp["cool"] = fuzz.trimf(temp.universe, [16, 20, 23])
        temp["neutral"] = fuzz.trimf(temp.universe, [21, 23.5, 26])
        temp["warm"] = fuzz.trimf(temp.universe, [24, 28, 32])
        temp["hot"] = fuzz.trimf(temp.universe, [30, 45, 45])

        humidity["dry"] = fuzz.trimf(humidity.universe, [0, 0, 35])
        humidity["comfortable"] = fuzz.trimf(humidity.universe, [25, 50, 65])
        humidity["humid"] = fuzz.trimf(humidity.universe, [55, 100, 100])

        comfort["very_cold"] = fuzz.trimf(comfort.universe, [-3, -3, -1.5])
        comfort["cold"] = fuzz.trimf(comfort.universe, [-2.5, -1.5, -0.5])
        comfort["neutral"] = fuzz.trimf(comfort.universe, [-1, 0, 1])
        comfort["warm"] = fuzz.trimf(comfort.universe, [0.5, 1.5, 2.5])
        comfort["hot"] = fuzz.trimf(comfort.universe, [1.5, 3, 3])

        rules = [
            ctrl.Rule(temp["cold"] & humidity["dry"], comfort["very_cold"]),
            ctrl.Rule(temp["cold"] & humidity["comfortable"], comfort["cold"]),
            ctrl.Rule(temp["cold"] & humidity["humid"], comfort["cold"]),
            ctrl.Rule(temp["cool"] & humidity["dry"], comfort["cold"]),
            ctrl.Rule(temp["cool"] & humidity["comfortable"], comfort["neutral"]),
            ctrl.Rule(temp["cool"] & humidity["humid"], comfort["neutral"]),
            ctrl.Rule(temp["neutral"] & humidity["dry"], comfort["neutral"]),
            ctrl.Rule(temp["neutral"] & humidity["comfortable"], comfort["neutral"]),
            ctrl.Rule(temp["neutral"] & humidity["humid"], comfort["warm"]),
            ctrl.Rule(temp["warm"] & humidity["dry"], comfort["warm"]),
            ctrl.Rule(temp["warm"] & humidity["comfortable"], comfort["warm"]),
            ctrl.Rule(temp["warm"] & humidity["humid"], comfort["hot"]),
            ctrl.Rule(temp["hot"] & humidity["dry"], comfort["warm"]),
            ctrl.Rule(temp["hot"] & humidity["comfortable"], comfort["hot"]),
            ctrl.Rule(temp["hot"] & humidity["humid"], comfort["hot"]),
        ]
        self._system = ctrl.ControlSystemSimulation(ctrl.ControlSystem(rules))

    def train(self, features: np.ndarray, target: np.ndarray) -> None:
        preds = self.predict(features)
        valid = ~np.isnan(preds)
        self._metrics = {
            "rmse": float(np.sqrt(mean_squared_error(target[valid], preds[valid]))),
            "mae": float(mean_absolute_error(target[valid], preds[valid])),
            "r2": float(r2_score(target[valid], preds[valid])),
        }

    def predict(self, features: np.ndarray) -> np.ndarray:
        results = np.full(len(features), np.nan)
        for i, row in enumerate(features):
            try:
                self._system.input["temperature"] = np.clip(row[0], 5, 45)
                self._system.input["humidity"] = np.clip(row[1], 0, 100)
                self._system.compute()
                results[i] = self._system.output["comfort"]
            except Exception:
                pass
        return results

    def get_metrics(self) -> dict[str, float]:
        return self._metrics
