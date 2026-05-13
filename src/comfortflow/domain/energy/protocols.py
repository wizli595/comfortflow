from __future__ import annotations

from typing import Protocol

import numpy as np


class EnergyModel(Protocol):
    def train(self, features: np.ndarray, target: np.ndarray) -> None: ...
    def predict(self, features: np.ndarray) -> np.ndarray: ...
    def get_metrics(self) -> dict[str, float]: ...
