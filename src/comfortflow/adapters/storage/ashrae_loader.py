"""Load and clean ASHRAE DB II CSV into training-ready DataFrames."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

# Map column index -> clean name (indices from the CSV header)
COLUMN_INDEX_MAP = {
    14: "thermal_sensation",
    29: "air_temperature_celsius",
    49: "relative_humidity_percent",
    52: "air_velocity_ms",
    23: "clothing_insulation_clo",
    24: "metabolic_rate_met",
    5: "climate",
    8: "building_type",
    3: "season",
    7: "country",
}

NUMERIC_COLS = [
    "thermal_sensation",
    "air_temperature_celsius",
    "relative_humidity_percent",
    "air_velocity_ms",
    "clothing_insulation_clo",
    "metabolic_rate_met",
]

FEATURE_COLS = [
    "air_temperature_celsius",
    "relative_humidity_percent",
    "air_velocity_ms",
    "clothing_insulation_clo",
    "metabolic_rate_met",
]


def load_ashrae(path: Path) -> pd.DataFrame:
    raw = pd.read_csv(path, encoding="latin-1", low_memory=False)
    df = _select_columns(raw)
    df = _coerce_numeric(df)
    df = _drop_incomplete(df)
    df = _remove_outliers(df)
    return df.reset_index(drop=True)


def split_features_target(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    return df[FEATURE_COLS].values, df["thermal_sensation"].values


def _select_columns(raw: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(
        {name: raw.iloc[:, idx] for idx, name in COLUMN_INDEX_MAP.items()}
    )


def _coerce_numeric(df: pd.DataFrame) -> pd.DataFrame:
    for col in NUMERIC_COLS:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def _drop_incomplete(df: pd.DataFrame) -> pd.DataFrame:
    return df.dropna(subset=NUMERIC_COLS)


def _remove_outliers(df: pd.DataFrame) -> pd.DataFrame:
    return df[
        df["air_temperature_celsius"].between(5, 45)
        & df["relative_humidity_percent"].between(0, 100)
        & df["air_velocity_ms"].between(0, 5)
        & df["clothing_insulation_clo"].between(0, 3)
        & df["metabolic_rate_met"].between(0.5, 4)
        & df["thermal_sensation"].between(-3, 3)
    ]
