"""Use case: train a comfort model on gold/ data."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from comfortflow.domain.comfort.protocols import ComfortModel, FeatureRepository


@dataclass
class TrainResult:
    model_name: str
    metrics: dict[str, float]


def train_comfort_model(
    model: ComfortModel,
    repository: FeatureRepository,
    building_id: str = "all",
) -> TrainResult:
    features = repository.get_comfort_features(building_id)
    targets = repository.get_comfort_targets(building_id)
    model.train(features, targets)
    return TrainResult(model_name=type(model).__name__, metrics=model.get_metrics())
