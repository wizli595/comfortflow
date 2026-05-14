"""Wire adapters into use cases (dependency injection)."""

from __future__ import annotations

from functools import lru_cache

from comfortflow.adapters.ml.xgboost_comfort import XGBoostComfortModel
from comfortflow.adapters.storage.parquet_repository import ParquetFeatureRepository


@lru_cache
def get_repository() -> ParquetFeatureRepository:
    return ParquetFeatureRepository()


@lru_cache
def get_comfort_model() -> XGBoostComfortModel:
    repo = get_repository()
    model = XGBoostComfortModel()
    model.train(repo.get_comfort_features(), repo.get_comfort_targets())
    return model
