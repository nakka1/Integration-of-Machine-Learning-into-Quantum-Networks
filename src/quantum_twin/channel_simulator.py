"""
Component 1 -- `WDMChannelSimulator`.

Generates synthetic time series via an Ornstein-Uhlenbeck process and
derives the latent quantum fidelity F(t). Logic identical to the original
prototype; only extracted into an importable, testable module.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import MinMaxScaler


class WDMChannelSimulator:
    """
    Synthetic simulator of a WDM (Wavelength Division Multiplexing) optical
    channel operating at the network edge.

    Generates two continuous physical variables via Ornstein-Uhlenbeck (OU)
    processes:

        - phase_deviation : classical optical signal phase deviation (rad, >= 0)
        - temp_gradient   : local temperature gradient (K/m, >= 0)

    The latent quantum fidelity F(t) is derived from the phase deviation:
    inversely proportional to it, plus Gaussian noise, clipped to [0, 1].
    """

    def __init__(self, n_steps: int = 4000, dt: float = 0.01, seed: int = 42) -> None:
        self.n_steps = n_steps
        self.dt = dt
        self.rng = np.random.default_rng(seed)

    def _ornstein_uhlenbeck(self, theta: float, mu: float, sigma: float, x0: float) -> np.ndarray:
        """
        Numerically integrates (Euler-Maruyama) an Ornstein-Uhlenbeck process:
            dX_t = theta * (mu - X_t) * dt + sigma * dW_t
        """
        x = np.zeros(self.n_steps, dtype=np.float64)
        x[0] = x0
        sqrt_dt = np.sqrt(self.dt)
        for t in range(1, self.n_steps):
            dW = self.rng.normal(0.0, sqrt_dt)
            x[t] = x[t - 1] + theta * (mu - x[t - 1]) * self.dt + sigma * dW
        return x

    def generate_dataset(self) -> pd.DataFrame:
        """Generates the full synthetic dataset (features + latent fidelity)."""
        phase_deviation = self._ornstein_uhlenbeck(theta=0.70, mu=0.30, sigma=0.15, x0=0.30)
        phase_deviation = np.abs(phase_deviation)

        temp_gradient = self._ornstein_uhlenbeck(theta=0.50, mu=0.50, sigma=0.10, x0=0.50)
        temp_gradient = np.abs(temp_gradient)

        alpha = 1.4
        eps = self.rng.normal(0.0, 0.03, self.n_steps)
        fidelity = 1.0 - alpha * phase_deviation + eps
        fidelity = np.clip(fidelity, 0.0, 1.0)

        return pd.DataFrame({
            "phase_deviation": phase_deviation,
            "temp_gradient": temp_gradient,
            "fidelity": fidelity,
        })

    def preprocess(self, df: pd.DataFrame, window_size: int = 20,
                    test_size: float = 0.2) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor,
                                                       torch.Tensor, MinMaxScaler]:
        """
        Normalizes features with MinMaxScaler, builds sliding windows
        (batch, seq_len, n_features), and splits train/test without
        shuffling (preserves chronological order).
        """
        features = df[["phase_deviation", "temp_gradient"]].values
        target = df[["fidelity"]].values

        feat_scaler = MinMaxScaler(feature_range=(0.0, 1.0))
        features_scaled = feat_scaler.fit_transform(features)

        X, y = [], []
        for i in range(len(features_scaled) - window_size):
            X.append(features_scaled[i:i + window_size])
            y.append(target[i + window_size])
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.float32)

        split_idx = int(len(X) * (1.0 - test_size))
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]

        X_train_t = torch.tensor(X_train, dtype=torch.float32)
        y_train_t = torch.tensor(y_train, dtype=torch.float32)
        X_test_t = torch.tensor(X_test, dtype=torch.float32)
        y_test_t = torch.tensor(y_test, dtype=torch.float32)

        return X_train_t, y_train_t, X_test_t, y_test_t, feat_scaler
