"""LSTM energy predictor implementing EnergyModel protocol."""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

from comfortflow.config import get_config


class _LSTMNet(nn.Module):
    def __init__(self, input_dim: int, hidden: int = 128) -> None:
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden, num_layers=2, batch_first=True, dropout=0.1)
        self.fc = nn.Sequential(nn.Linear(hidden, 64), nn.ReLU(), nn.Linear(64, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])


class LSTMEnergyModel:
    def __init__(self, seq_len: int = 12, epochs: int = 50, lr: float = 0.001) -> None:
        self._net: _LSTMNet | None = None
        self._x_scaler = StandardScaler()
        self._y_scaler = StandardScaler()
        self._metrics: dict[str, float] = {}
        self._seq_len = seq_len
        self._epochs = epochs
        self._lr = lr

    def train(self, features: np.ndarray, target: np.ndarray) -> None:
        X_s = self._x_scaler.fit_transform(features)
        y_s = self._y_scaler.fit_transform(target.reshape(-1, 1)).flatten()

        X_seq, y_seq = self._make_episode_sequences(X_s, y_s)
        split = int(len(X_seq) * 0.9)

        dl = DataLoader(
            TensorDataset(torch.FloatTensor(X_seq[:split]), torch.FloatTensor(y_seq[:split])),
            batch_size=512, shuffle=True,
        )
        self._net = _LSTMNet(features.shape[1])
        opt = torch.optim.Adam(self._net.parameters(), lr=self._lr)
        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=5, factor=0.5)

        for _ in range(self._epochs):
            self._net.train()
            total = 0.0
            for xb, yb in dl:
                opt.zero_grad()
                loss = nn.MSELoss()(self._net(xb).squeeze(), yb)
                loss.backward()
                opt.step()
                total += loss.item()
            sched.step(total / len(dl))

        self._net.eval()
        with torch.no_grad():
            val_p = self._net(torch.FloatTensor(X_seq[split:])).numpy().flatten()
        val_real = self._y_scaler.inverse_transform(val_p.reshape(-1, 1)).flatten()
        y_real = self._y_scaler.inverse_transform(y_seq[split:].reshape(-1, 1)).flatten()
        self._metrics = {
            "rmse": float(np.sqrt(mean_squared_error(y_real, val_real))),
            "mae": float(mean_absolute_error(y_real, val_real)),
            "r2": float(r2_score(y_real, val_real)),
        }

    def predict(self, features: np.ndarray) -> np.ndarray:
        X_s = self._x_scaler.transform(features)
        X_seq, _ = self._make_episode_sequences(X_s, np.zeros(len(features)))
        self._net.eval()
        with torch.no_grad():
            p = self._net(torch.FloatTensor(X_seq)).numpy().flatten()
        return self._y_scaler.inverse_transform(p.reshape(-1, 1)).flatten()

    def get_metrics(self) -> dict[str, float]:
        return self._metrics

    def _make_episode_sequences(self, X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Build sequences without crossing episode boundaries."""
        episode_len = len(X) // max(1, self._detect_episodes(X))
        X_seq, y_seq = [], []
        for start in range(0, len(X) - episode_len + 1, episode_len):
            ep_X = X[start:start + episode_len]
            ep_y = y[start:start + episode_len]
            for i in range(len(ep_X) - self._seq_len):
                X_seq.append(ep_X[i:i + self._seq_len])
                y_seq.append(ep_y[i + self._seq_len])
        return np.array(X_seq), np.array(y_seq)

    @staticmethod
    def _detect_episodes(X: np.ndarray) -> int:
        """Detect number of episodes by counting resets in the first column (month)."""
        diffs = np.diff(X[:, 0])
        resets = np.sum(np.abs(diffs) > 5)
        return max(1, resets + 1)
