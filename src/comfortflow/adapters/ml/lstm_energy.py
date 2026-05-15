"""LSTM energy predictor implementing EnergyModel protocol."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler


class _LSTMNet(nn.Module):
    def __init__(self, input_dim: int, hidden: int = 64, layers: int = 2) -> None:
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden, layers, batch_first=True, dropout=0.2)
        self.fc = nn.Sequential(nn.Linear(hidden, 32), nn.ReLU(), nn.Linear(32, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])


class LSTMEnergyModel:
    def __init__(self, seq_len: int = 24, epochs: int = 50, lr: float = 0.001) -> None:
        self._net: _LSTMNet | None = None
        self._scaler = StandardScaler()
        self._metrics: dict[str, float] = {}
        self._seq_len = seq_len
        self._epochs = epochs
        self._lr = lr

    def train(self, features: np.ndarray, target: np.ndarray) -> None:
        X_scaled = self._scaler.fit_transform(features)
        X_seq, y_seq = self._make_sequences(X_scaled, target)

        split = int(len(X_seq) * 0.9)
        X_tr, X_val = X_seq[:split], X_seq[split:]
        y_tr, y_val = y_seq[:split], y_seq[split:]

        self._net = _LSTMNet(features.shape[1])
        optimizer = torch.optim.Adam(self._net.parameters(), lr=self._lr)
        loss_fn = nn.MSELoss()

        X_t = torch.FloatTensor(X_tr)
        y_t = torch.FloatTensor(y_tr).unsqueeze(1)

        for epoch in range(self._epochs):
            self._net.train()
            optimizer.zero_grad()
            loss = loss_fn(self._net(X_t), y_t)
            loss.backward()
            optimizer.step()

        self._net.eval()
        with torch.no_grad():
            val_preds = self._net(torch.FloatTensor(X_val)).numpy().flatten()
        self._metrics = {
            "rmse": float(np.sqrt(mean_squared_error(y_val, val_preds))),
            "mae": float(mean_absolute_error(y_val, val_preds)),
            "r2": float(r2_score(y_val, val_preds)),
        }

    def predict(self, features: np.ndarray) -> np.ndarray:
        X_scaled = self._scaler.transform(features)
        X_seq, _ = self._make_sequences(X_scaled, np.zeros(len(features)))
        self._net.eval()
        with torch.no_grad():
            return self._net(torch.FloatTensor(X_seq)).numpy().flatten()

    def get_metrics(self) -> dict[str, float]:
        return self._metrics

    def _make_sequences(self, X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        X_seq, y_seq = [], []
        for i in range(len(X) - self._seq_len):
            X_seq.append(X[i : i + self._seq_len])
            y_seq.append(y[i + self._seq_len])
        return np.array(X_seq), np.array(y_seq)
