import numpy as np
import pytest
import torch

from quantum_twin.baselines import (
    MovingAveragePredictor,
    OraclePredictor,
    PersistencePredictor,
    build_moving_average_baseline,
    build_oracle_baseline,
    build_persistence_baseline,
)


# ---------------------------------------------------------------------------
# PersistencePredictor
# ---------------------------------------------------------------------------

def test_persistence_full_batch_matches_shifted_sequence():
    y_train = torch.tensor([0.9, 0.85, 0.80])
    y_test = torch.tensor([0.75, 0.70, 0.65, 0.60, 0.55])

    predictor = build_persistence_baseline(y_train, y_test)
    predictor.eval()
    out = predictor(torch.zeros(5, 1))

    expected = torch.tensor([0.80, 0.75, 0.70, 0.65, 0.60]).reshape(-1, 1)
    assert torch.allclose(out, expected, atol=1e-6)


def test_persistence_step_by_step_matches_full_batch():
    y_train = torch.tensor([0.9, 0.85, 0.80])
    y_test = torch.tensor([0.75, 0.70, 0.65, 0.60, 0.55])

    full_batch_predictor = build_persistence_baseline(y_train, y_test)
    full_batch_predictor.eval()
    full_batch_out = full_batch_predictor(torch.zeros(5, 1)).reshape(-1)

    step_predictor = build_persistence_baseline(y_train, y_test)
    step_predictor.eval()
    step_out = torch.cat([step_predictor(torch.zeros(1, 1)) for _ in range(5)]).reshape(-1)

    assert torch.allclose(full_batch_out, step_out, atol=1e-6)


def test_persistence_eval_resets_cursor_for_a_second_pass():
    predictor = PersistencePredictor(torch.tensor([0.1, 0.2, 0.3]), warm_start=0.5)
    predictor.eval()
    first_pass = predictor(torch.zeros(3, 1)).clone()
    predictor.eval()  # reset
    second_pass = predictor(torch.zeros(3, 1))
    assert torch.allclose(first_pass, second_pass)


def test_persistence_predictions_are_clipped_to_unit_interval():
    predictor = PersistencePredictor(torch.tensor([1.5, -0.3, 0.5]), warm_start=0.9)
    predictor.eval()
    out = predictor(torch.zeros(3, 1))
    assert out.min() >= 0.0 and out.max() <= 1.0


def test_persistence_to_and_eval_return_self():
    predictor = PersistencePredictor(torch.tensor([0.5, 0.6]))
    assert predictor.to("cpu") is predictor
    assert predictor.eval() is predictor


# ---------------------------------------------------------------------------
# MovingAveragePredictor
# ---------------------------------------------------------------------------

def test_moving_average_matches_manual_computation():
    y_train = torch.tensor([0.9, 0.85, 0.80])
    y_test = torch.tensor([0.75, 0.70, 0.65, 0.60, 0.55])

    predictor = build_moving_average_baseline(y_train, y_test, window=2)
    predictor.eval()
    out = predictor(torch.zeros(5, 1)).reshape(-1)

    # idx0: warm_start = mean(y_train[-2:]) = mean(0.85, 0.80) = 0.825
    # idx1: mean(y_test[0:1]) = 0.75
    # idx2: mean(y_test[0:2]) = mean(0.75, 0.70) = 0.725
    # idx3: mean(y_test[1:3]) = mean(0.70, 0.65) = 0.675
    # idx4: mean(y_test[2:4]) = mean(0.65, 0.60) = 0.625
    expected = torch.tensor([0.825, 0.75, 0.725, 0.675, 0.625])
    assert torch.allclose(out, expected, atol=1e-6)


def test_moving_average_window_one_equals_persistence():
    y_train = torch.tensor([0.9])
    y_test = torch.tensor([0.75, 0.70, 0.65])

    ma = build_moving_average_baseline(y_train, y_test, window=1)
    ma.eval()
    ma_out = ma(torch.zeros(3, 1))

    persistence = build_persistence_baseline(y_train, y_test)
    persistence.eval()
    p_out = persistence(torch.zeros(3, 1))

    assert torch.allclose(ma_out, p_out, atol=1e-6)


# ---------------------------------------------------------------------------
# OraclePredictor
# ---------------------------------------------------------------------------

def test_oracle_predicts_true_fidelity_exactly():
    y_test = torch.tensor([0.75, 0.70, 0.65, 0.60, 0.55])
    predictor = build_oracle_baseline(y_test)
    predictor.eval()
    out = predictor(torch.zeros(5, 1)).reshape(-1)
    assert torch.allclose(out, y_test, atol=1e-6)


def test_oracle_out_of_range_index_clamps_to_last_value():
    predictor = OraclePredictor(torch.tensor([0.5, 0.6, 0.7]))
    predictor.eval()
    # Ask for more predictions than the reference sequence has -- must
    # clamp to the last value rather than raising an IndexError.
    out = predictor(torch.zeros(5, 1)).reshape(-1)
    assert out[-1] == pytest.approx(0.7, abs=1e-6)


# ---------------------------------------------------------------------------
# Interface parity with trainable predictors (nn.Module contract)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("cls,kwargs", [
    (PersistencePredictor, {}),
    (MovingAveragePredictor, {}),
    (OraclePredictor, {}),
])
def test_naive_predictors_expose_eval_to_and_call_like_a_module(cls, kwargs):
    y_ref = torch.tensor([0.5, 0.6, 0.7, 0.8])
    predictor = cls(y_ref, **kwargs)
    assert predictor.eval() is predictor
    assert predictor.to(torch.device("cpu")) is predictor
    out = predictor(torch.zeros(4, 1))
    assert out.shape == (4, 1)
