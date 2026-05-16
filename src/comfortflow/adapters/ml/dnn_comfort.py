"""Deep Neural Network implementation of ComfortModel protocol."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler


class _DNNNet(nn.Module):
    def __init__(self, input_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128), nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(128, 64), nn.BatchNorm1d(64), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(64, 32), nn.BatchNorm1d(32), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(32, 16), nn.ReLU(),
            nn.Linear(16, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class DNNComfortModel:
    def __init__(self, epochs: int = 200, lr: float = 0.0005) -> None:
        self._net: _DNNNet | None = None
        self._scaler = StandardScaler()
        self._metrics: dict[str, float] = {}
        self._epochs = epochs
        self._lr = lr

    def train(self, features: np.ndarray, target: np.ndarray) -> None:
        X = self._scaler.fit_transform(features)
        X_t = torch.FloatTensor(X)
        y_t = torch.FloatTensor(target).unsqueeze(1)

        self._net = _DNNNet(X.shape[1])
        optimizer = torch.optim.Adam(self._net.parameters(), lr=self._lr)

        self._net.train()
        for _ in range(self._epochs):
            optimizer.zero_grad()
            nn.MSELoss()(self._net(X_t), y_t).backward()
            optimizer.step()

        self._net.eval()
        with torch.no_grad():
            preds = self._net(X_t).numpy().flatten()
        self._metrics = {
            "rmse": float(np.sqrt(mean_squared_error(target, preds))),
            "mae": float(mean_absolute_error(target, preds)),
            "r2": float(r2_score(target, preds)),
        }

    def predict(self, features: np.ndarray) -> np.ndarray:
        self._net.eval()
        X = torch.FloatTensor(self._scaler.transform(features))
        with torch.no_grad():
            return self._net(X).numpy().flatten()

    def get_metrics(self) -> dict[str, float]:
        return self._metrics
