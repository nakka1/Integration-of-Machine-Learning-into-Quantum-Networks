"""Tests for quantum_twin.metrics.quantum (mirrors src/quantum_twin/metrics/quantum.py)."""

import numpy as np
import pytest


def _step_degradation(n: int, drop_at: int, shift: float = 0.0, slope: float = 0.04) -> np.ndarray:
    """Synthetic fidelity trajectory that starts at 1.0 and linearly decays
    to 0.0 starting at time step `drop_at + shift`."""
    t = np.arange(n)
    return np.clip(1.0 - slope * np.maximum(t - drop_at - shift, 0), 0.0, 1.0)


def _baseline_metrics():
    return {"total_steps": 780, "useful_pairs": 300, "halted": 0,
            "attempted": 780, "avg_classical_latency_s": 0.0}




def _intelligent_metrics():
    return {"total_steps": 780, "useful_pairs": 280, "halted": 300,
            "attempted": 480, "avg_classical_latency_s": 0.00012}


# ---------------------------------------------------------------------------
# compute_regression_metrics / evaluate_predictor_regression


# ---------------------------------------------------------------------------
# compute_qpu_economy
# ---------------------------------------------------------------------------

from quantum_twin.metrics import compute_fidelity_statistics, compute_qpu_economy


def test_compute_qpu_economy_reports_positive_savings_when_halting():
    economy = compute_qpu_economy(_intelligent_metrics(), _baseline_metrics(), shots_per_attempt=512)

    assert economy["qpu_cycles_saved"] == 300
    assert economy["qpu_cycles_saved_pct"] == pytest.approx(300 / 780 * 100.0)
    assert economy["qpu_shots_saved"] == 300 * 512
    assert economy["useful_pairs_deficit_surplus"] == -20


def test_compute_qpu_economy_without_shots_leaves_shots_saved_none():
    economy = compute_qpu_economy(_intelligent_metrics(), _baseline_metrics())
    assert economy["qpu_shots_saved"] is None




# ---------------------------------------------------------------------------
# compute_fidelity_statistics
# ---------------------------------------------------------------------------

def test_compute_fidelity_statistics_basic_shape():
    y_true = _step_degradation(60, drop_at=25, shift=0.0)
    stats = compute_fidelity_statistics(y_true, threshold=0.65)

    assert 0.0 <= stats["mean_fidelity"] <= 1.0
    assert stats["min_fidelity"] <= stats["mean_fidelity"] <= stats["max_fidelity"]
    assert stats["n_degradation_events"] == 1
    assert 0.0 <= stats["pct_below_threshold"] <= 100.0


def test_compute_fidelity_statistics_empty_raises():
    with pytest.raises(ValueError):
        compute_fidelity_statistics([])


def test_compute_fidelity_statistics_constant_series_no_events():
    y_true = np.ones(20) * 0.95
    stats = compute_fidelity_statistics(y_true, threshold=0.65)
    assert stats["n_degradation_events"] == 0
    assert stats["n_recovery_events"] == 0
    assert stats["pct_below_threshold"] == pytest.approx(0.0)
