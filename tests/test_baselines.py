import pytest
import torch

from qrepeater_twin import EdgeLSTM, TinyTransformer
from qrepeater_twin.baselines import (
    RandomForestFidelityModel,
    train_lstm_mse,
    train_random_forest,
    train_transformer,
    train_xgboost,
)


def test_train_lstm_mse_reduces_loss():
    torch.manual_seed(0)
    X = torch.rand(16, 20, 2)
    y = torch.rand(16, 1)

    model = EdgeLSTM(input_size=2, hidden_size=8, num_layers=1)
    criterion = torch.nn.MSELoss()
    with torch.no_grad():
        loss_before = criterion(model(X), y).item()

    model = train_lstm_mse(model, X, y, epochs=50, lr=1e-2, seed=0)

    with torch.no_grad():
        loss_after = criterion(model(X), y).item()

    assert loss_after < loss_before


def test_random_forest_fits_and_predicts_in_unit_range():
    torch.manual_seed(0)
    X_train = torch.rand(32, 20, 2)
    y_train = torch.rand(32, 1)

    rf_model = RandomForestFidelityModel(n_estimators=20, max_depth=4, seed=0)
    rf_model.fit(X_train, y_train)
    orchestrator_model = rf_model.as_orchestrator_model()

    x_sample = torch.rand(1, 20, 2)
    orchestrator_model.eval()
    pred = orchestrator_model(x_sample)

    assert pred.shape == (1, 1)
    assert torch.all(pred >= 0.0) and torch.all(pred <= 1.0)


def test_train_random_forest_convenience_function():
    torch.manual_seed(0)
    X_train = torch.rand(32, 20, 2)
    y_train = torch.rand(32, 1)

    model = train_random_forest(X_train, y_train, n_estimators=10, max_depth=3, seed=0)
    pred = model(torch.rand(1, 20, 2))
    assert pred.shape == (1, 1)


def test_train_xgboost_skips_cleanly_when_not_installed():
    pytest.importorskip("xgboost", reason="xgboost is an optional dependency for this baseline")
    torch.manual_seed(0)
    X_train = torch.rand(32, 20, 2)
    y_train = torch.rand(32, 1)
    model = train_xgboost(X_train, y_train, n_estimators=10, max_depth=3, seed=0)
    pred = model(torch.rand(1, 20, 2))
    assert pred.shape == (1, 1)


def test_tiny_transformer_output_shape_and_range():
    torch.manual_seed(0)
    model = TinyTransformer(input_size=2, d_model=8, nhead=2, num_layers=1, dim_feedforward=16)
    x = torch.rand(4, 20, 2)
    out = model(x)
    assert out.shape == (4, 1)
    assert torch.all(out >= 0.0) and torch.all(out <= 1.0)


def test_train_transformer_reduces_loss():
    torch.manual_seed(0)
    X = torch.rand(16, 20, 2)
    y = torch.rand(16, 1)

    model = TinyTransformer(input_size=2, d_model=8, nhead=2, num_layers=1, dim_feedforward=16)
    criterion = torch.nn.MSELoss()
    with torch.no_grad():
        loss_before = criterion(model(X), y).item()

    model = train_transformer(model, X, y, epochs=60, lr=5e-3, seed=0)

    with torch.no_grad():
        loss_after = criterion(model(X), y).item()

    assert loss_after < loss_before
