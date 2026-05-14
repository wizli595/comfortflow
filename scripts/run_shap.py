"""Run SHAP analysis on XGBoost comfort model."""

from __future__ import annotations

import shap
import numpy as np

from comfortflow.adapters.ml.xgboost_comfort import XGBoostComfortModel
from comfortflow.adapters.storage.parquet_repository import ParquetFeatureRepository, FEATURE_COLS


def main() -> None:
    repo = ParquetFeatureRepository()
    model = XGBoostComfortModel()
    model.train(repo.get_comfort_features(), repo.get_comfort_targets())

    explainer = shap.TreeExplainer(model._model)
    sample = repo.test_features[:1000]
    shap_values = explainer.shap_values(sample)

    importance = np.abs(shap_values).mean(axis=0)
    ranked = sorted(zip(FEATURE_COLS, importance), key=lambda x: x[1], reverse=True)

    print("=== SHAP Feature Importance ===")
    for name, score in ranked:
        bar = "#" * int(score * 20)
        print(f"  {name:35s} {score:.4f}  {bar}")


if __name__ == "__main__":
    main()
