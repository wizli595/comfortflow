"""Train all comfort models and compare in MLflow."""

from __future__ import annotations

import os
import sys

import mlflow
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from comfortflow.adapters.storage.parquet_repository import ParquetFeatureRepository
from comfortflow.application.train_comfort_model import train_comfort_model

# Fix Windows emoji encoding
sys.stdout.reconfigure(encoding="utf-8")
os.environ["PYTHONIOENCODING"] = "utf-8"


def evaluate_on_test(model, repo: ParquetFeatureRepository) -> dict[str, float]:
    preds = model.predict(repo.test_features)
    valid = ~np.isnan(preds)
    return {
        "test_rmse": float(np.sqrt(mean_squared_error(repo.test_targets[valid], preds[valid]))),
        "test_mae": float(mean_absolute_error(repo.test_targets[valid], preds[valid])),
        "test_r2": float(r2_score(repo.test_targets[valid], preds[valid])),
    }


def train_and_log(name: str, model, repo: ParquetFeatureRepository) -> None:
    print(f"\n--- {name} ---")
    with mlflow.start_run(run_name=name):
        result = train_comfort_model(model, repo)
        test_metrics = evaluate_on_test(model, repo)
        mlflow.log_metrics({**result.metrics, **test_metrics})
        mlflow.log_param("model", name)
        print(f"  Val:  RMSE={result.metrics['rmse']:.4f}  MAE={result.metrics['mae']:.4f}  R2={result.metrics['r2']:.4f}")
        print(f"  Test: RMSE={test_metrics['test_rmse']:.4f}  MAE={test_metrics['test_mae']:.4f}  R2={test_metrics['test_r2']:.4f}")


def main() -> None:
    mlflow.set_tracking_uri("http://localhost:5000")
    mlflow.set_experiment("comfort-prediction")
    repo = ParquetFeatureRepository()

    # XGBoost
    from comfortflow.adapters.ml.xgboost_comfort import XGBoostComfortModel
    train_and_log("xgboost", XGBoostComfortModel(), repo)

    # Random Forest
    from comfortflow.adapters.ml.random_forest_comfort import RandomForestComfortModel
    train_and_log("random-forest", RandomForestComfortModel(), repo)

    # SVM (slow — uses 20K sample)
    from comfortflow.adapters.ml.svm_comfort import SVMComfortModel
    train_and_log("svm", SVMComfortModel(), repo)

    # ANN
    from comfortflow.adapters.ml.ann_comfort import ANNComfortModel
    train_and_log("ann", ANNComfortModel(), repo)

    # DNN
    from comfortflow.adapters.ml.dnn_comfort import DNNComfortModel
    train_and_log("dnn", DNNComfortModel(), repo)

    # Fuzzy Logic
    from comfortflow.adapters.ml.fuzzy_comfort import FuzzyComfortModel
    train_and_log("fuzzy-logic", FuzzyComfortModel(), repo)

    print("\n=== All models trained. Check MLflow: http://localhost:5000 ===")


if __name__ == "__main__":
    main()
