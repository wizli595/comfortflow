import numpy as np

from comfortflow.application.train_comfort_model import train_comfort_model


class FakeRepository:
    def get_comfort_features(self, building_id="all"):
        return np.random.rand(100, 5)

    def get_comfort_targets(self, building_id="all"):
        return np.random.rand(100)


class FakeModel:
    def __init__(self):
        self.trained = False

    def train(self, features, target):
        self.trained = True

    def predict(self, features):
        return np.zeros(len(features))

    def get_metrics(self):
        return {"rmse": 0.5, "mae": 0.3, "r2": 0.7}


class TestTrainComfortModel:
    def test_returns_metrics(self) -> None:
        result = train_comfort_model(FakeModel(), FakeRepository())
        assert result.metrics["rmse"] == 0.5

    def test_model_gets_trained(self) -> None:
        model = FakeModel()
        train_comfort_model(model, FakeRepository())
        assert model.trained
