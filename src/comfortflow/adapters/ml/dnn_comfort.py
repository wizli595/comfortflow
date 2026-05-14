"""Deep Neural Network implementation of ComfortModel protocol."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
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
        X_tr, X_val, y_tr, y_val = train_test_split(features, target, test_size=0.1, random_state=42)
        X_tr = self._scaler.fit_transform(X_tr)
        X_val = self._scaler.transform(X_val)
        X_t = torch.FloatTensor(X_tr)
        y_t = torch.FloatTensor(y_tr).unsqueeze(1)
        self._net = _DNNNet(X_tr.shape[1])
        optimizer = torch.optim.Adam(self._net.parameters(), lr=self._lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5)
        loss_fn = nn.MSELoss()

        best_val_loss = float("inf")
        patience, wait = 15, 0
        for _ in range(self._epochs):
            self._net.train()
            optimizer.zero_grad()
            loss = loss_fn(self._net(X_t), y_t)
            loss.backward()
            optimizer.step()

            self._net.eval()
            with torch.no_grad():
                val_preds = self._net(torch.FloatTensor(X_val)).numpy().flatten()
            val_loss = mean_squared_error(y_val, val_preds)
            scheduler.step(val_loss)
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                wait = 0
            else:
                wait += 1
                if wait >= patience:
                    break

        self._metrics = {
            "rmse": float(np.sqrt(mean_squared_error(y_val, val_preds))),
            "mae": float(mean_absolute_error(y_val, val_preds)),
            "r2": float(r2_score(y_val, val_preds)),
        }

    def predict(self, features: np.ndarray) -> np.ndarray:
        self._net.eval()
        X = torch.FloatTensor(self._scaler.transform(features))
        with torch.no_grad():
            return self._net(X).numpy().flatten()

    def get_metrics(self) -> dict[str, float]:
        return self._metrics
