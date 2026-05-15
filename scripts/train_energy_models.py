"""Train energy prediction models and log to MLflow."""

from __future__ import annotations

import os
import sys

import mlflow
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from comfortflow.adapters.storage.energy_repository import EnergyRepository

sys.stdout.reconfigure(encoding="utf-8")
os.environ["PYTHONIOENCODING"] = "utf-8"


def evaluate(model, repo: EnergyRepository) -> dict[str, float]:
    preds = model.predict(repo.test_features)
    n = min(len(preds), len(repo.test_targets))
    return {
        "test_rmse": float(np.sqrt(mean_squared_error(repo.test_targets[:n], preds[:n]))),
        "test_mae": float(mean_absolute_error(repo.test_targets[:n], preds[:n])),
        "test_r2": float(r2_score(repo.test_targets[:n], preds[:n])),
    }


def train_and_log(name: str, model, repo: EnergyRepository) -> None:
    print(f"\n--- {name} ---")
    with mlflow.start_run(run_name=name):
        model.train(repo.get_energy_features(), repo.get_energy_targets())
        val_metrics = model.get_metrics()
        test_metrics = evaluate(model, repo)
        mlflow.log_metrics({**val_metrics, **test_metrics})
        mlflow.log_param("model", name)
        print(f"  Val:  RMSE={val_metrics['rmse']:.2f}  MAE={val_metrics['mae']:.2f}  R2={val_metrics['r2']:.4f}")
        print(f"  Test: RMSE={test_metrics['test_rmse']:.2f}  MAE={test_metrics['test_mae']:.2f}  R2={test_metrics['test_r2']:.4f}")


def main() -> None:
    mlflow.set_tracking_uri("http://localhost:5000")
    mlflow.set_experiment("energy-prediction")
    repo = EnergyRepository()

    from comfortflow.adapters.ml.xgboost_energy import XGBoostEnergyModel
    train_and_log("xgboost-energy", XGBoostEnergyModel(), repo)

    from comfortflow.adapters.ml.lstm_energy import LSTMEnergyModel
    train_and_log("lstm-energy", LSTMEnergyModel(), repo)

    print("\n=== Energy models trained. Check MLflow: http://localhost:5000 ===")


if __name__ == "__main__":
    main()
