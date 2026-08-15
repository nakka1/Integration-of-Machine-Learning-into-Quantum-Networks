"""Tests for quantum_twin.metrics.performance (mirrors src/quantum_twin/metrics/performance.py)."""

import pytest

from quantum_twin.config import EnergyConfig
from quantum_twin.metrics import compute_energy_report, compute_latency_ratio, compute_throughput


def _baseline_metrics():
    return {"total_steps": 780, "useful_pairs": 300, "halted": 0,
            "attempted": 780, "avg_classical_latency_s": 0.0}




def _intelligent_metrics():
    return {"total_steps": 780, "useful_pairs": 280, "halted": 300,
            "attempted": 480, "avg_classical_latency_s": 0.00012}


# ---------------------------------------------------------------------------
# compute_regression_metrics / evaluate_predictor_regression


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
def test_compute_throughput_positive_and_bounded_by_blind():
    baseline = compute_throughput(_baseline_metrics(), cycle_time_s=1e-3)
    intelligent = compute_throughput(_intelligent_metrics(), cycle_time_s=1e-3)

    assert baseline["throughput_pairs_per_s"] > 0
    assert intelligent["throughput_pairs_per_s"] > 0
    # The predictive controller adds classical latency on every cycle
    # (blind never pays it), so for equal cycle_time_s its total_time_s
    # is >= the blind baseline's.
    assert intelligent["total_time_s"] >= baseline["total_time_s"]


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
