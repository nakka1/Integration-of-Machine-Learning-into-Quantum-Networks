import pytest

from qrepeater_twin.config import EnergyConfig
from qrepeater_twin.metrics import (
    build_decision_matrix,
    compute_energy_report,
    compute_qpu_economy,
    compute_throughput,
)


def _baseline_metrics():
    return {"total_steps": 780, "useful_pairs": 300, "halted": 0,
            "attempted": 780, "avg_classical_latency_s": 0.0}


def _intelligent_metrics():
    return {"total_steps": 780, "useful_pairs": 280, "halted": 300,
            "attempted": 480, "avg_classical_latency_s": 0.00012}


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


def test_build_decision_matrix_ranks_dominant_model_first():
    rows = [
        {
            "Model": "Blind",
            "qpu_yield_pct": 38.46,
            "throughput_pairs_per_s": 384.6,
            "qpu_cycles_saved_pct": 0.0,
            "energy_saved_pct": 0.0,
            "inference_latency_ms": 0.0,
        },
        {
            "Model": "Dominant",
            "qpu_yield_pct": 90.0,
            "throughput_pairs_per_s": 500.0,
            "qpu_cycles_saved_pct": 50.0,
            "energy_saved_pct": 40.0,
            "inference_latency_ms": 0.01,
        },
    ]
    weights = {
        "qpu_yield_pct": 0.25,
        "throughput_pairs_per_s": 0.20,
        "qpu_cycles_saved_pct": 0.20,
        "energy_saved_pct": 0.20,
        "inference_latency_ms": 0.15,
    }

    dm = build_decision_matrix(rows, weights)

    assert dm.iloc[0]["Model"] == "Dominant"
    assert dm.iloc[0]["Rank"] == 1
    assert dm.iloc[0]["Decision Score"] > dm.iloc[1]["Decision Score"]


def test_build_decision_matrix_missing_criterion_raises():
    rows = [{"Model": "A", "qpu_yield_pct": 50.0}]
    weights = {"qpu_yield_pct": 0.5, "throughput_pairs_per_s": 0.5}
    with pytest.raises(KeyError):
        build_decision_matrix(rows, weights)


def test_build_decision_matrix_empty_rows_returns_empty_frame():
    dm = build_decision_matrix([], {"qpu_yield_pct": 1.0})
    assert dm.empty
