import math

import pytest
import torch

from qrepeater_twin.config import EnergyConfig
from qrepeater_twin.metrics import (
    build_decision_matrix,
    compute_confusion_metrics,
    compute_energy_report,
    compute_latency_ratio,
    compute_qpu_economy,
    compute_regression_metrics,
    compute_throughput,
    evaluate_predictor_regression,
)


def _baseline_metrics():
    return {"total_steps": 780, "useful_pairs": 300, "halted": 0,
            "attempted": 780, "avg_classical_latency_s": 0.0}


def _intelligent_metrics():
    return {"total_steps": 780, "useful_pairs": 280, "halted": 300,
            "attempted": 480, "avg_classical_latency_s": 0.00012}


# ---------------------------------------------------------------------------
# compute_regression_metrics / evaluate_predictor_regression
# ---------------------------------------------------------------------------

def test_compute_regression_metrics_matches_manual_computation():
    y_true = [0.9, 0.8, 0.5, 0.3, 0.95]
    y_pred = [0.85, 0.82, 0.55, 0.25, 0.90]

    result = compute_regression_metrics(y_true, y_pred)

    errors = [p - t for p, t in zip(y_pred, y_true)]
    mae_expected = sum(abs(e) for e in errors) / len(errors)
    rmse_expected = math.sqrt(sum(e ** 2 for e in errors) / len(errors))

    assert result["mae"] == pytest.approx(mae_expected)
    assert result["rmse"] == pytest.approx(rmse_expected)
    assert 0.0 < result["r2"] <= 1.0


def test_compute_regression_metrics_perfect_prediction():
    y = [0.9, 0.8, 0.5, 0.3, 0.95]
    result = compute_regression_metrics(y, y)
    assert result["mae"] == pytest.approx(0.0, abs=1e-12)
    assert result["rmse"] == pytest.approx(0.0, abs=1e-12)
    assert result["r2"] == pytest.approx(1.0)


def test_compute_regression_metrics_constant_y_true_gives_nan_r2():
    result = compute_regression_metrics([0.5, 0.5, 0.5], [0.4, 0.5, 0.6])
    assert math.isnan(result["r2"])


def test_compute_regression_metrics_accepts_torch_tensors():
    y_true = torch.tensor([[0.9], [0.8], [0.5]])
    y_pred = torch.tensor([[0.85], [0.82], [0.55]])
    result = compute_regression_metrics(y_true, y_pred)
    assert result["mae"] > 0.0


def test_compute_regression_metrics_shape_mismatch_raises():
    with pytest.raises(ValueError):
        compute_regression_metrics([0.1, 0.2], [0.1, 0.2, 0.3])


def test_evaluate_predictor_regression_uses_model_forward_pass():
    torch.manual_seed(0)

    class _ConstantPredictor:
        """Minimal stand-in for a trained model: always predicts 0.7."""

        def eval(self):
            return self

        def __call__(self, x):
            return torch.full((x.shape[0], 1), 0.7)

    X_test = torch.rand(10, 20, 2)
    y_test = torch.full((10, 1), 0.7)

    result = evaluate_predictor_regression(_ConstantPredictor(), X_test, y_test)
    assert result["mae"] == pytest.approx(0.0, abs=1e-6)
    assert result["r2"] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# compute_confusion_metrics
# ---------------------------------------------------------------------------

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
# ---------------------------------------------------------------------------

def test_compute_latency_ratio_matches_manual_division():
    ratio = compute_latency_ratio(1e-4, 30e-6)
    assert ratio == pytest.approx(1e-4 / 30e-6)


def test_compute_latency_ratio_is_dimensionless_and_scale_invariant():
    # Scaling both tau_inf and T2 by the same factor leaves C_latencia unchanged.
    r1 = compute_latency_ratio(1e-4, 30e-6)
    r2 = compute_latency_ratio(2e-4, 60e-6)
    assert r1 == pytest.approx(r2)


def test_compute_latency_ratio_zero_t2_raises():
    with pytest.raises(ValueError):
        compute_latency_ratio(1e-4, 0.0)


def test_compute_latency_ratio_negative_t2_raises():
    with pytest.raises(ValueError):
        compute_latency_ratio(1e-4, -30e-6)


# ---------------------------------------------------------------------------
# compute_throughput / compute_qpu_economy / compute_energy_report
# ---------------------------------------------------------------------------

def test_compute_throughput_positive_and_bounded_by_blind():
    baseline = compute_throughput(_baseline_metrics(), cycle_time_s=1e-3)
    intelligent = compute_throughput(_intelligent_metrics(), cycle_time_s=1e-3)

    assert baseline["throughput_pairs_per_s"] > 0
    assert intelligent["throughput_pairs_per_s"] > 0
    # The predictive controller adds classical latency on every cycle
    # (blind never pays it), so for equal cycle_time_s its total_time_s
    # is >= the blind baseline's.
    assert intelligent["total_time_s"] >= baseline["total_time_s"]


def test_compute_qpu_economy_reports_positive_savings_when_halting():
    economy = compute_qpu_economy(_intelligent_metrics(), _baseline_metrics(), shots_per_attempt=512)

    assert economy["qpu_cycles_saved"] == 300
    assert economy["qpu_cycles_saved_pct"] == pytest.approx(300 / 780 * 100.0)
    assert economy["qpu_shots_saved"] == 300 * 512
    assert economy["useful_pairs_deficit_surplus"] == -20


def test_compute_qpu_economy_without_shots_leaves_shots_saved_none():
    economy = compute_qpu_economy(_intelligent_metrics(), _baseline_metrics())
    assert economy["qpu_shots_saved"] is None


def test_compute_energy_report_zero_cycles_saved_means_equal_quantum_energy():
    # A "predictor" that never halts (attempted == baseline attempted)
    # must show identical *quantum*-side energy to the baseline; only the
    # classical inference term can differ.
    always_purify = dict(_intelligent_metrics())
    always_purify["attempted"] = 780
    always_purify["halted"] = 0

    report = compute_energy_report(always_purify, _baseline_metrics(), shots=512)
    baseline_quantum_only = compute_energy_report(_baseline_metrics(), _baseline_metrics(), shots=512)
    assert report["quantum_energy_j"] == pytest.approx(baseline_quantum_only["quantum_energy_j"])


def test_compute_energy_report_custom_energy_config_scales_linearly():
    cfg_low = EnergyConfig(joules_per_1q_gate=1e-9, joules_per_2q_gate=1e-9, joules_per_shot_overhead=0.0,
                            classical_inference_power_w=0.0, classical_idle_power_w=0.0)
    cfg_high = EnergyConfig(joules_per_1q_gate=2e-9, joules_per_2q_gate=2e-9, joules_per_shot_overhead=0.0,
                             classical_inference_power_w=0.0, classical_idle_power_w=0.0)

    metrics = _intelligent_metrics()
    baseline = _baseline_metrics()

    report_low = compute_energy_report(metrics, baseline, shots=512, energy_cfg=cfg_low)
    report_high = compute_energy_report(metrics, baseline, shots=512, energy_cfg=cfg_high)

    assert report_high["quantum_energy_j"] == pytest.approx(2 * report_low["quantum_energy_j"])


# ---------------------------------------------------------------------------
# build_decision_matrix (latency_ratio_c as the cost-type latency criterion)
# ---------------------------------------------------------------------------

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

