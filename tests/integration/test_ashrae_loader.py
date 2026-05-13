from pathlib import Path

import pytest

from comfortflow.adapters.storage.ashrae_loader import load_ashrae, split_features_target

DATA_PATH = Path("data/ashrae_db2.01.csv")


@pytest.mark.skipif(not DATA_PATH.exists(), reason="ASHRAE CSV not found")
class TestASHRAELoader:
    def test_loads_and_cleans(self) -> None:
        df = load_ashrae(DATA_PATH)
        assert len(df) > 50_000
        assert df["thermal_sensation"].between(-3, 3).all()
        assert df["air_velocity_ms"].max() <= 5
        numeric = ["thermal_sensation", "air_temperature_celsius",
                   "relative_humidity_percent", "air_velocity_ms",
                   "clothing_insulation_clo", "metabolic_rate_met"]
        assert df[numeric].isna().sum().sum() == 0

    def test_split_features_target(self) -> None:
        df = load_ashrae(DATA_PATH)
        X, y = split_features_target(df)
        assert X.shape[0] == y.shape[0]
        assert X.shape[1] == 5
        assert y.min() >= -3 and y.max() <= 3
