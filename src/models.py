"""
Component 2 -- `EdgeLSTM` and `CS_MSELoss` (parameterized for the sweep).

`CS_MSELoss` exposes `lambda_penalty` as the main hyperparameter of the
Pareto Frontier, instantiable dynamically: `CS_MSELoss(lambda_penalty=L)`.
The remaining terms (a moderate False Negative penalty and excess-discard
regularization) remain as stabilizers that prevent the model from
trivially collapsing at any point of the lambda sweep.

`train_edge_lstm` explicitly accepts a `seed` and calls `torch.manual_seed`
before any parameter initialization -- a prerequisite for the multi-seed
averaging implemented in `pareto_sweep.py` (each seed must produce a
genuinely different, reproducible weight initialization).
"""

from __future__ import annotations

import torch
import torch.nn as nn


class EdgeLSTM(nn.Module):
    """
    Lightweight recurrent neural network ("Edge LSTM"), designed for fast
    inference on resource-constrained edge hardware.

    Architecture:
        input   -> (batch, seq_len, 2)  [phase_deviation, temp_gradient]
        LSTM    -> compact hidden_size, few layers
        output  -> linear layer + sigmoid, producing F_hat(t) in [0, 1]
    """

    def __init__(self, input_size: int = 2, hidden_size: int = 16, num_layers: int = 1):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
        )
        self.head = nn.Linear(hidden_size, 1)
        self.activation = nn.Sigmoid()  # ensures F_hat(t) is in [0, 1]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        last_hidden = out[:, -1, :]
        pred = self.activation(self.head(last_hidden))
        return pred


class CS_MSELoss(nn.Module):
    """
    Cost-Sensitive Mean Squared Error (CS-MSE), parameterized as the "knob"
    of the Pareto Frontier.

    Terms:
        - lambda_penalty : SEVERE penalty on False Positives (F_true < threshold
                            <= F_pred). This is the hyperparameter swept over
                            in the optimization loop -- the higher it is, the
                            more conservative the model, the higher the QPU
                            efficiency, and the lower the throughput.
        - lambda_fn       : MODERATE penalty on False Negatives (F_pred <
                            threshold <= F_true), kept fixed during the sweep
                            to avoid the model fully collapsing into
                            "discard everything" at high lambda_penalty
                            values.
        - discard_penalty_weight / max_discard_rate : batch-level
          regularization that penalizes discard rates above
          max_discard_rate, reinforcing training stability across the
          whole sweep.
    """

    def __init__(self, threshold: float = 0.65, lambda_penalty: float = 10.0,
                 lambda_fn: float = 2.0, discard_penalty_weight: float = 5.0,
                 max_discard_rate: float = 0.60):
        super().__init__()
        self.threshold = threshold
        self.lambda_penalty = lambda_penalty
        self.lambda_fn = lambda_fn
        self.discard_penalty_weight = discard_penalty_weight
        self.max_discard_rate = max_discard_rate

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        squared_error = (y_pred - y_true) ** 2

        is_false_positive = (y_true < self.threshold) & (y_pred >= self.threshold)
        is_false_negative = (y_true >= self.threshold) & (y_pred < self.threshold)

        weights = torch.ones_like(squared_error)
        weights = torch.where(is_false_positive, torch.full_like(squared_error, self.lambda_penalty), weights)
        weights = torch.where(is_false_negative, torch.full_like(squared_error, self.lambda_fn), weights)

        weighted_mse = (squared_error * weights).mean()

        # Excess-discard penalty (batch level), differentiable via sigmoid.
        soft_discard_indicator = torch.sigmoid((self.threshold - y_pred) * 50.0)
        discard_rate = soft_discard_indicator.mean()
        excess_discard = torch.clamp(discard_rate - self.max_discard_rate, min=0.0)
        discard_penalty = self.discard_penalty_weight * (excess_discard ** 2)

        return weighted_mse + discard_penalty


def train_edge_lstm(model: nn.Module, X_train: torch.Tensor, y_train: torch.Tensor,
                     threshold: float = 0.65, lambda_penalty: float = 10.0, lambda_fn: float = 2.0,
                     discard_penalty_weight: float = 5.0, max_discard_rate: float = 0.60,
                     epochs: int = 120, lr: float = 3e-3, device: torch.device = None,
                     seed: int = None, verbose: bool = False):
    """
    Single-batch (full-batch) training routine (compact dataset).

    `device` determines where the model (and implicitly the tensors, which
    must already be on the same device) is trained. Kept explicit to allow
    consistent GPU/CPU handling throughout the pipeline.

    When provided, `seed` is re-applied via `torch.manual_seed` at the start
    of this function, for reproducibility of the optimization loop itself
    (Adam's internal operation order, etc.). **Important**: since training
    here is full-batch (no DataLoader/shuffling), the only real source of
    variation between seeds is the model's weight initialization -- and
    that initialization must already have happened *before* this call,
    with the same seed applied immediately before `EdgeLSTM(...)` is
    instantiated. This is exactly the pattern (seed -> build model -> train)
    that `pareto_sweep.run_pareto_sweep` follows on every round of the
    multi-seed averaging, ensuring each round starts from a genuinely
    different, reproducible initialization.
    """
    if seed is not None:
        torch.manual_seed(seed)

    if device is not None:
        model = model.to(device)

    criterion = CS_MSELoss(threshold=threshold, lambda_penalty=lambda_penalty, lambda_fn=lambda_fn,
                            discard_penalty_weight=discard_penalty_weight,
                            max_discard_rate=max_discard_rate)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        y_pred = model(X_train)
        loss = criterion(y_pred, y_train)
        loss.backward()
        optimizer.step()
        if verbose and (epoch + 1) % 30 == 0:
            with torch.no_grad():
                discard_rate_now = (y_pred < threshold).float().mean().item()
            print(f"    Epoch {epoch + 1:3d}/{epochs} | CS-MSE Loss: {loss.item():.6f} | "
                  f"Discard rate (train): {discard_rate_now*100:.1f}%")
    return model
