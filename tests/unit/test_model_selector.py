import numpy as np

from comfortflow.adapters.ml.model_selector import ModelSelector


class GoodModel:
    def predict(self, features):
        return np.zeros(len(features))


class BadModel:
    def predict(self, features):
        return np.ones(len(features)) * 5


class TestModelSelector:
    def test_picks_best(self) -> None:
        selector = ModelSelector()
        selector.register("good", GoodModel())
        selector.register("bad", BadModel())
        selector.evaluate(np.random.rand(50, 5), np.zeros(50))
        assert selector.best_model_name() == "good"

    def test_rankings_sorted(self) -> None:
        selector = ModelSelector()
        selector.register("good", GoodModel())
        selector.register("bad", BadModel())
        selector.evaluate(np.random.rand(50, 5), np.zeros(50))
        rankings = selector.rankings()
        assert rankings[0][0] == "good"
        assert rankings[0][1] < rankings[1][1]
