"""Train all comfort models with proper train/test split in application layer."""

from __future__ import annotations

import os
import sys

import mlflow
import numpy as np
from sklearn.metrics import accuracy_score, mean_absolute_error, mean_squared_error, r2_score

from comfortflow.adapters.storage.parquet_repository import ParquetFeatureRepository
from comfortflow.application.train_comfort_model import train_comfort_model

sys.stdout.reconfigure(encoding="utf-8")
os.environ["PYTHONIOENCODING"] = "utf-8"


def evaluate(model, X_test: np.ndarray, y_test: np.ndarray) -> dict[str, float]:
    preds = model.predict(X_test)
    valid = ~np.isnan(preds)
    p, t = preds[valid], y_test[valid]
    pred_cls = np.clip(np.round(p), -3, 3).astype(int)
    true_cls = np.clip(np.round(t), -3, 3).astype(int)
    return {
        "test_rmse": float(np.sqrt(mean_squared_error(t, p))),
        "test_mae": float(mean_absolute_error(t, p)),
        "test_r2": float(r2_score(t, p)),
        "test_accuracy": float(accuracy_score(true_cls, pred_cls)),
        "test_within_1": float(np.mean(np.abs(pred_cls - true_cls) <= 1)),
    }


def train_and_log(name: str, model, repo: ParquetFeatureRepository) -> None:
    print(f"\n--- {name} ---")
    with mlflow.start_run(run_name=name):
        result, X_test, y_test = train_comfort_model(model, repo)
        test_metrics = evaluate(model, X_test, y_test)
        mlflow.log_metrics(test_metrics)
        mlflow.log_param("model", name)
        print(f"  RMSE={test_metrics['test_rmse']:.4f}  R2={test_metrics['test_r2']:.4f}  "
              f"Acc={test_metrics['test_accuracy']:.1%}  W1={test_metrics['test_within_1']:.1%}")


def main() -> None:
    mlflow.set_tracking_uri("http://localhost:5000")
    mlflow.set_experiment("comfort-prediction-v2")
    repo = ParquetFeatureRepository()

    from comfortflow.adapters.ml.xgboost_comfort import XGBoostComfortModel
    train_and_log("xgboost", XGBoostComfortModel(), repo)

    from comfortflow.adapters.ml.random_forest_comfort import RandomForestComfortModel
    train_and_log("random-forest", RandomForestComfortModel(), repo)

    from comfortflow.adapters.ml.svm_comfort import SVMComfortModel
    train_and_log("svm", SVMComfortModel(), repo)

    from comfortflow.adapters.ml.ann_comfort import ANNComfortModel
    train_and_log("ann", ANNComfortModel(), repo)

    from comfortflow.adapters.ml.dnn_comfort import DNNComfortModel
    train_and_log("dnn", DNNComfortModel(), repo)

    from comfortflow.adapters.ml.fuzzy_comfort import FuzzyComfortModel
    train_and_log("fuzzy-logic", FuzzyComfortModel(), repo)

    print("\n=== Done. Check MLflow: http://localhost:5000 ===")


if __name__ == "__main__":
    main()
