"""Train XGBoost with 6 features (including radiant temperature)."""

from __future__ import annotations

import sys
import os

import mlflow
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import accuracy_score, mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

sys.stdout.reconfigure(encoding="utf-8")
os.environ["PYTHONIOENCODING"] = "utf-8"


def main() -> None:
    raw = pd.read_csv("data/ashrae_db2.01.csv", encoding="latin-1", low_memory=False)
    df = extract_6_features(raw)
    print(f"Rows with radiant temp: {len(df):,}")

    X = df[["air_temp", "radiant_temp", "humidity", "velocity", "clo", "met"]].values
    y = df["sensation"].values
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)

    model = xgb.XGBRegressor(
        n_estimators=300, max_depth=6, learning_rate=0.1,
        subsample=0.8, colsample_bytree=0.8, random_state=42,
    )
    model.fit(X_tr, y_tr, eval_set=[(X_te, y_te)], verbose=False)
    preds = model.predict(X_te)

    rmse = float(np.sqrt(mean_squared_error(y_te, preds)))
    mae = float(mean_absolute_error(y_te, preds))
    r2 = float(r2_score(y_te, preds))
    pred_cls = np.clip(np.round(preds), -3, 3).astype(int)
    true_cls = np.clip(np.round(y_te), -3, 3).astype(int)
    acc = float(accuracy_score(true_cls, pred_cls))
    within_1 = float(np.mean(np.abs(pred_cls - true_cls) <= 1))

    mlflow.set_tracking_uri("http://localhost:5000")
    mlflow.set_experiment("comfort-prediction")
    with mlflow.start_run(run_name="xgboost-6features"):
        mlflow.log_params({"model": "xgboost-6features", "features": 6, "rows": len(df)})
        mlflow.log_metrics({"rmse": rmse, "mae": mae, "r2": r2, "accuracy": acc, "within_1": within_1})

    print(f"RMSE={rmse:.4f}  MAE={mae:.4f}  R2={r2:.4f}")
    print(f"Accuracy={acc:.1%}  Within 1 vote={within_1:.1%}")
    print()
    print("Comparison:")
    print(f"  5 features (77K rows): R2=0.220  Acc=42.5%  W1=83.4%")
    print(f"  6 features (30K rows): R2={r2:.3f}  Acc={acc:.1%}  W1={within_1:.1%}")


def extract_6_features(raw: pd.DataFrame) -> pd.DataFrame:
    df = pd.DataFrame({
        "air_temp": pd.to_numeric(raw.iloc[:, 29], errors="coerce"),
        "radiant_temp": pd.to_numeric(raw.iloc[:, 39], errors="coerce"),
        "humidity": pd.to_numeric(raw.iloc[:, 49], errors="coerce"),
        "velocity": pd.to_numeric(raw.iloc[:, 52], errors="coerce"),
        "clo": pd.to_numeric(raw.iloc[:, 23], errors="coerce"),
        "met": pd.to_numeric(raw.iloc[:, 24], errors="coerce"),
        "sensation": pd.to_numeric(raw.iloc[:, 14], errors="coerce"),
    })
    df = df.dropna()
    return df[
        df["air_temp"].between(5, 45) & df["radiant_temp"].between(5, 50)
        & df["humidity"].between(0, 100) & df["velocity"].between(0, 5)
        & df["clo"].between(0, 3) & df["met"].between(0.5, 4)
        & df["sensation"].between(-3, 3)
    ].reset_index(drop=True)


if __name__ == "__main__":
    main()
