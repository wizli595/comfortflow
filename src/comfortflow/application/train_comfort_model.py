"""Use case: train a comfort model on gold/ data."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.model_selection import train_test_split

from comfortflow.config import get_config
from comfortflow.domain.comfort.protocols import ComfortModel, FeatureRepository


@dataclass
class TrainResult:
    model_name: str
    metrics: dict[str, float]


def train_comfort_model(
    model: ComfortModel,
    repository: FeatureRepository,
    building_id: str = "all",
) -> tuple[TrainResult, np.ndarray, np.ndarray]:
    cfg = get_config()["split"]
    features = repository.get_comfort_features(building_id)
    targets = repository.get_comfort_targets(building_id)

    X_train, X_test, y_train, y_test = train_test_split(
        features, targets,
        test_size=cfg["test_size"],
        random_state=cfg["random_state"],
    )
    model.train(X_train, y_train)
    return TrainResult(model_name=type(model).__name__, metrics=model.get_metrics()), X_test, y_test
