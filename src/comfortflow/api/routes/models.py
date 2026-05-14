"""Model comparison routes."""

from __future__ import annotations

from fastapi import APIRouter

import mlflow

router = APIRouter(prefix="/models", tags=["models"])


@router.get("/compare")
def compare_models():
    mlflow.set_tracking_uri("http://localhost:5000")
    experiment = mlflow.get_experiment_by_name("comfort-prediction")
    if not experiment:
        return {"error": "No experiment found"}

    runs = mlflow.search_runs(experiment_ids=[experiment.experiment_id])
    results = []
    for _, run in runs.iterrows():
        results.append({
            "model": run.get("params.model", "unknown"),
            "test_rmse": run.get("metrics.test_rmse"),
            "test_mae": run.get("metrics.test_mae"),
            "test_r2": run.get("metrics.test_r2"),
        })

    return sorted(results, key=lambda x: x.get("test_rmse") or 999)
