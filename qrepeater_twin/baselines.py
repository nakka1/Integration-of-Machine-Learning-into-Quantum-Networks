"""
Component 5 -- Predictor baselines compared against the intelligent
admission controller (EdgeLSTM + CS_MSELoss).

Every baseline here ends up wrapped so it exposes the exact same runtime
contract the rest of the pipeline already relies on
(`orchestrator.DigitalTwinOrchestrator.run_intelligent`):

    - `.eval()`                       (no-op for non-torch models)
    - `model(x)` with `x` of shape (1, window_size, n_features)
      returning something with a scalar `.item()` in [0, 1]

This lets `DigitalTwinOrchestrator` -- and therefore `InferenceTimer`,
`QuantumRepeaterNode.apply_latency_decay`, and every downstream metric --
run *unmodified* regardless of whether the underlying predictor is a
recurrent network, a tree ensemble, or a Transformer. Comparability
across architectures was the whole point of adding these baselines: they
all get admitted into the digital twin loop through the same door.

Baselines implemented:

    1. LSTM + MSE        : `train_lstm_mse` -- the *same* `EdgeLSTM`
                             architecture as the intelligent controller,
                             trained with a plain, cost-insensitive
                             `nn.MSELoss`. Isolates the contribution of
                             `CS_MSELoss` itself (architecture held fixed).
    2. Random Forest      : `RandomForestFidelityModel` -- classical,
                             non-recurrent ensemble over the flattened
                             window.
    3. XGBoost            : `XGBoostFidelityModel` -- gradient-boosted
                             trees over the flattened window. Optional
                             dependency: raises a clear, catchable
                             `ImportError` (not a hard crash) if `xgboost`
                             isn't installed, so callers can skip it.
    4. Transformer        : `TinyTransformer` + `train_transformer` -- a
                             small Transformer encoder over the same input
                             window, trained with plain `nn.MSELoss`, as a
                             higher-capacity architectural baseline.
"""

from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# 1) LSTM + plain MSE (same architecture as EdgeLSTM, cost-insensitive loss)
# ---------------------------------------------------------------------------

def train_lstm_mse(model: nn.Module, X_train: torch.Tensor, y_train: torch.Tensor,
                    epochs: int = 150, lr: float = 0.012, device: torch.device = None,
                    seed: int = None, verbose: bool = False) -> nn.Module:
    """
    Trains an `EdgeLSTM` (or any compatible module) with a plain
    `nn.MSELoss`, i.e. WITHOUT the False-Positive-severe, cost-sensitive
    weighting of `CS_MSELoss`.

    Kept architecturally identical to `models.train_edge_lstm` (same
    full-batch loop, same seed-before-init convention) so that any
    difference observed downstream (QPU yield, useful pairs, latency) is
    attributable to the *loss function*, not to incidental differences in
    the training procedure.
    """
    if seed is not None:
        torch.manual_seed(seed)

    if device is not None:
        model = model.to(device)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        y_pred = model(X_train)
        loss = criterion(y_pred, y_train)
        loss.backward()
        optimizer.step()
        if verbose and (epoch + 1) % 30 == 0:
            print(f"    [LSTM+MSE] Epoch {epoch + 1:3d}/{epochs} | MSE Loss: {loss.item():.6f}")
    return model


# ---------------------------------------------------------------------------
# Shared helpers: window flattening for the non-recurrent baselines
# ---------------------------------------------------------------------------

def _flatten_windows(X: torch.Tensor) -> np.ndarray:
    """(batch, seq_len, n_features) -> (batch, seq_len * n_features), on CPU/numpy."""
    X_np = X.detach().cpu().numpy() if isinstance(X, torch.Tensor) else np.asarray(X)
    return X_np.reshape(X_np.shape[0], -1)


class _SklearnRegressorAdapter:
    """
    Wraps a fitted scikit-learn-style regressor (`.predict(X)`) so it can
    be dropped into `DigitalTwinOrchestrator.run_intelligent` exactly like
    a `torch.nn.Module`: same `.eval()` / callable contract, same
    (1, 1)-shaped, [0, 1]-clipped tensor output.

    These models have no notion of `torch` autograd or CUDA, so
    `InferenceTimer` in `run_intelligent` transparently falls back to
    `perf_counter()` for them (its CUDA-Events branch is only entered when
    `device.type == "cuda"`, and this adapter always returns a CPU
    tensor) -- still a fair, isolated latency measurement of exactly the
    `.predict(...)` call, nothing else.
    """

    def __init__(self, fitted_estimator, name: str):
        self._estimator = fitted_estimator
        self.name = name

    def eval(self) -> "_SklearnRegressorAdapter":
        return self  # stateless at inference time; kept for interface parity

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        x_flat = _flatten_windows(x)
        pred = self._estimator.predict(x_flat)
        pred = np.clip(pred, 0.0, 1.0)
        return torch.as_tensor(pred, dtype=torch.float32).reshape(-1, 1)


# ---------------------------------------------------------------------------
# 2) Random Forest
# ---------------------------------------------------------------------------

class RandomForestFidelityModel:
    """
    Random Forest regressor baseline for F_hat(t), trained on the
    flattened (window_size * n_features) feature vector.

    Unlike `EdgeLSTM`, this model has no notion of sequence order beyond
    whatever the tree splits pick up from the flattened positions -- a
    useful contrast against the recurrent baselines: does explicit
    temporal structure (LSTM) actually earn its keep over an
    order-agnostic ensemble with the same raw information?
    """

    def __init__(self, n_estimators: int = 200, max_depth: int = 8, seed: int = 42):
        from sklearn.ensemble import RandomForestRegressor

        self.model = RandomForestRegressor(
            n_estimators=n_estimators, max_depth=max_depth,
            random_state=seed, n_jobs=-1,
        )

    def fit(self, X_train: torch.Tensor, y_train: torch.Tensor) -> "RandomForestFidelityModel":
        X_flat = _flatten_windows(X_train)
        y_flat = _flatten_windows(y_train).ravel()
        self.model.fit(X_flat, y_flat)
        return self

    def as_orchestrator_model(self) -> _SklearnRegressorAdapter:
        """Wraps the fitted estimator for use with `DigitalTwinOrchestrator`."""
        return _SklearnRegressorAdapter(self.model, name="RandomForest")


def train_random_forest(X_train: torch.Tensor, y_train: torch.Tensor,
                         n_estimators: int = 200, max_depth: int = 8,
                         seed: int = 42) -> _SklearnRegressorAdapter:
    """Convenience one-shot: fit a Random Forest and return it orchestrator-ready."""
    rf = RandomForestFidelityModel(n_estimators=n_estimators, max_depth=max_depth, seed=seed)
    rf.fit(X_train, y_train)
    return rf.as_orchestrator_model()


# ---------------------------------------------------------------------------
# 3) XGBoost (optional dependency)
# ---------------------------------------------------------------------------

class XGBoostFidelityModel:
    """
    Gradient-boosted trees (XGBoost) baseline for F_hat(t), trained on the
    same flattened window representation as `RandomForestFidelityModel`.

    `xgboost` is an OPTIONAL dependency (not in `requirements.txt` by
    default): the import is attempted lazily, inside `__init__`, and
    raises a plain `ImportError` with an actionable message
    (`pip install xgboost`) rather than crashing the whole comparison run.
    Callers (see `model_comparison.run_model_comparison`) catch this and
    skip the XGBoost row, logging a warning instead of failing the sweep.
    """

    def __init__(self, n_estimators: int = 200, max_depth: int = 5,
                 learning_rate: float = 0.1, seed: int = 42):
        try:
            from xgboost import XGBRegressor
        except ImportError as exc:  # pragma: no cover - exercised only when xgboost is absent
            raise ImportError(
                "XGBoost baseline requested but the 'xgboost' package is not installed. "
                "Install it with `pip install xgboost` or set "
                "ComparisonConfig.include_xgboost=False to skip this baseline."
            ) from exc

        self.model = XGBRegressor(
            n_estimators=n_estimators, max_depth=max_depth,
            learning_rate=learning_rate, objective="reg:squarederror",
            random_state=seed, n_jobs=-1,
        )

    def fit(self, X_train: torch.Tensor, y_train: torch.Tensor) -> "XGBoostFidelityModel":
        X_flat = _flatten_windows(X_train)
        y_flat = _flatten_windows(y_train).ravel()
        self.model.fit(X_flat, y_flat)
        return self

    def as_orchestrator_model(self) -> _SklearnRegressorAdapter:
        return _SklearnRegressorAdapter(self.model, name="XGBoost")


def train_xgboost(X_train: torch.Tensor, y_train: torch.Tensor,
                   n_estimators: int = 200, max_depth: int = 5,
                   learning_rate: float = 0.1, seed: int = 42) -> _SklearnRegressorAdapter:
    """Convenience one-shot: fit an XGBoost regressor and return it orchestrator-ready.

    Raises `ImportError` if `xgboost` is not installed -- see
    `XGBoostFidelityModel`.
    """
    xgb = XGBoostFidelityModel(n_estimators=n_estimators, max_depth=max_depth,
                                learning_rate=learning_rate, seed=seed)
    xgb.fit(X_train, y_train)
    return xgb.as_orchestrator_model()


# ---------------------------------------------------------------------------
# 4) Transformer encoder
# ---------------------------------------------------------------------------

class _PositionalEncoding(nn.Module):
    """Standard sinusoidal positional encoding (Vaswani et al., 2017)."""

    def __init__(self, d_model: int, max_len: int = 512):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32)
                              * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term[: pe[:, 1::2].shape[1]])
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1), :]


class TinyTransformer(nn.Module):
    """
    Small Transformer-encoder baseline for F_hat(t), operating on the same
    (batch, seq_len, n_features) windows as `EdgeLSTM`.

    Architecture:
        input projection (Linear: n_features -> d_model)
        -> sinusoidal positional encoding
        -> `num_layers` x TransformerEncoderLayer (self-attention + FFN)
        -> mean-pool over the sequence dimension
        -> linear head + sigmoid, producing F_hat(t) in [0, 1]

    A higher-capacity, attention-based architectural baseline against
    which the compact, recurrent `EdgeLSTM` (designed for cheap edge
    inference) can be compared on the yield/latency/energy trade-off,
    not just on raw predictive accuracy.
    """

    def __init__(self, input_size: int = 2, d_model: int = 32, nhead: int = 4,
                 num_layers: int = 2, dim_feedforward: int = 64, dropout: float = 0.1):
        super().__init__()
        self.input_proj = nn.Linear(input_size, d_model)
        self.pos_encoding = _PositionalEncoding(d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
            dropout=dropout, batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.head = nn.Linear(d_model, 1)
        self.activation = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.input_proj(x)
        h = self.pos_encoding(h)
        h = self.encoder(h)
        pooled = h.mean(dim=1)
        pred = self.activation(self.head(pooled))
        return pred


def train_transformer(model: nn.Module, X_train: torch.Tensor, y_train: torch.Tensor,
                       epochs: int = 150, lr: float = 0.005, device: torch.device = None,
                       seed: int = None, verbose: bool = False) -> nn.Module:
    """
    Trains a `TinyTransformer` with a plain `nn.MSELoss`, following the
    same full-batch, seed-before-init convention as `train_edge_lstm` /
    `train_lstm_mse` so results stay directly comparable.
    """
    if seed is not None:
        torch.manual_seed(seed)

    if device is not None:
        model = model.to(device)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        y_pred = model(X_train)
        loss = criterion(y_pred, y_train)
        loss.backward()
        optimizer.step()
        if verbose and (epoch + 1) % 30 == 0:
            print(f"    [Transformer] Epoch {epoch + 1:3d}/{epochs} | MSE Loss: {loss.item():.6f}")
    return model
