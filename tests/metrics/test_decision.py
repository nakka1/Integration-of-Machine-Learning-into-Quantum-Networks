"""Tests for quantum_twin.metrics.decision (mirrors src/quantum_twin/metrics/decision.py)."""

import math

import pytest

from quantum_twin.metrics import build_decision_matrix, compute_confusion_metrics


def test_compute_confusion_metrics_matches_manual_rates():
    confusion = {"TP": 80, "FP": 20, "TN": 150, "FN": 30}
    result = compute_confusion_metrics(confusion)

    assert result["precision"] == pytest.approx(80 / 100)
    assert result["recall"] == pytest.approx(80 / 110)
    assert result["specificity"] == pytest.approx(150 / 170)
    assert result["fpr"] == pytest.approx(20 / 170)
    assert result["fnr"] == pytest.approx(30 / 110)
    assert 0.0 < result["f1"] < 1.0


def test_compute_confusion_metrics_degenerate_zero_positives_precision_is_nan():
    # No admissions were ever predicted (TP == FP == 0): precision is
    # undefined (0/0), not zero.
    confusion = {"TP": 0, "FP": 0, "TN": 10, "FN": 0}
    result = compute_confusion_metrics(confusion)
    assert math.isnan(result["precision"])
    assert math.isnan(result["recall"])  # TP + FN == 0 too


def test_compute_confusion_metrics_blind_baseline_shape():
    # The blind baseline's degenerate confusion matrix: FN == TN == 0
    # always (unconditional admission -- see orchestrator.run_blind_baseline).
    confusion = {"TP": 300, "FP": 480, "TN": 0, "FN": 0}
    result = compute_confusion_metrics(confusion)
    assert result["recall"] == pytest.approx(1.0)  # no good photon ever discarded
    assert result["specificity"] == pytest.approx(0.0)  # TN=0, FP=480 -> 0/480
    assert result["fnr"] == pytest.approx(0.0)  # FN=0, TP=300 -> 0/300


# ---------------------------------------------------------------------------
# compute_latency_ratio (C_latencia = tau_inf / T2)
def test_build_decision_matrix_ranks_dominant_model_first():
    rows = [
        {
            "Model": "Blind",
            "qpu_yield_pct": 38.46,
            "throughput_pairs_per_s": 384.6,
            "qpu_cycles_saved_pct": 0.0,
            "energy_saved_pct": 0.0,
            "latency_ratio_c": 0.0,
        },
        {
            "Model": "Dominant",
            "qpu_yield_pct": 90.0,
            "throughput_pairs_per_s": 500.0,
            "qpu_cycles_saved_pct": 50.0,
            "energy_saved_pct": 40.0,
            "latency_ratio_c": 0.02,
        },
    ]
    weights = {
        "qpu_yield_pct": 0.25,
        "throughput_pairs_per_s": 0.20,
        "qpu_cycles_saved_pct": 0.20,
        "energy_saved_pct": 0.20,
        "latency_ratio_c": 0.15,
    }

    dm = build_decision_matrix(rows, weights)

    assert dm.iloc[0]["Model"] == "Dominant"
    assert dm.iloc[0]["Rank"] == 1
    assert dm.iloc[0]["Decision Score"] > dm.iloc[1]["Decision Score"]


def test_build_decision_matrix_latency_ratio_c_is_cost_type():
    # Two models identical except for latency_ratio_c: the one with the
    # SMALLER ratio (closer to zero, i.e. negligible next to T2) must win.
    rows = [
        {"Model": "LowLatency", "qpu_yield_pct": 50.0, "latency_ratio_c": 0.001},
        {"Model": "HighLatency", "qpu_yield_pct": 50.0, "latency_ratio_c": 0.5},
    ]
    weights = {"qpu_yield_pct": 0.5, "latency_ratio_c": 0.5}
    dm = build_decision_matrix(rows, weights)
    assert dm.iloc[0]["Model"] == "LowLatency"


def test_build_decision_matrix_missing_criterion_raises():
    rows = [{"Model": "A", "qpu_yield_pct": 50.0}]
    weights = {"qpu_yield_pct": 0.5, "throughput_pairs_per_s": 0.5}
    with pytest.raises(KeyError):
        build_decision_matrix(rows, weights)


def test_build_decision_matrix_empty_rows_returns_empty_frame():
    dm = build_decision_matrix([], {"qpu_yield_pct": 1.0})
    assert dm.empty

