"""Train XGBoost comfort model and log to MLflow."""

from __future__ import annotations

import mlflow
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from comfortflow.adapters.ml.xgboost_comfort import XGBoostComfortModel
from comfortflow.adapters.storage.parquet_repository import ParquetFeatureRepository
from comfortflow.application.train_comfort_model import train_comfort_model


def main() -> None:
    mlflow.set_tracking_uri("http://localhost:5000")
    mlflow.set_experiment("comfort-prediction")

    repo = ParquetFeatureRepository()
    model = XGBoostComfortModel()

    with mlflow.start_run(run_name="xgboost-baseline"):
        result = train_comfort_model(model, repo)

        # Evaluate on held-out test set
        predictions = model.predict(repo.test_features)
        test_metrics = {
            "test_rmse": float(np.sqrt(mean_squared_error(repo.test_targets, predictions))),
            "test_mae": float(mean_absolute_error(repo.test_targets, predictions)),
            "test_r2": float(r2_score(repo.test_targets, predictions)),
        }

        all_metrics = {**result.metrics, **test_metrics}
        mlflow.log_metrics(all_metrics)
        mlflow.log_param("model", result.model_name)
        mlflow.log_param("features", 5)
        mlflow.log_param("train_rows", len(repo.get_comfort_features()))
        mlflow.log_param("test_rows", len(repo.test_features))

        print(f"Model: {result.model_name}")
        print(f"Validation: RMSE={all_metrics['rmse']:.4f}  MAE={all_metrics['mae']:.4f}  R2={all_metrics['r2']:.4f}")
        print(f"Test:       RMSE={test_metrics['test_rmse']:.4f}  MAE={test_metrics['test_mae']:.4f}  R2={test_metrics['test_r2']:.4f}")
        print(f"Logged to MLflow: http://localhost:5000")


if __name__ == "__main__":
    main()
