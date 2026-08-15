import itertools

import pytest

from quantum_twin.sensitivity import run_weight_sensitivity_analysis, summarize_robustness


BASE_WEIGHTS = {
    "qpu_yield_pct": 0.25,
    "throughput_pairs_per_s": 0.20,
    "qpu_cycles_saved_pct": 0.20,
    "energy_saved_pct": 0.20,
    "latency_ratio_c": 0.15,
}


def _dominant_rows():
    return [
        {"Model": "Blind", "qpu_yield_pct": 38.46, "throughput_pairs_per_s": 384.6,
         "qpu_cycles_saved_pct": 0.0, "energy_saved_pct": 0.0, "latency_ratio_c": 0.0},
        {"Model": "LSTM+MSE", "qpu_yield_pct": 60.0, "throughput_pairs_per_s": 420.0,
         "qpu_cycles_saved_pct": 20.0, "energy_saved_pct": 10.0, "latency_ratio_c": 0.01},
        {"Model": "RandomForest", "qpu_yield_pct": 55.0, "throughput_pairs_per_s": 400.0,
         "qpu_cycles_saved_pct": 15.0, "energy_saved_pct": 5.0, "latency_ratio_c": 0.005},
        {"Model": "EdgeLSTM+CS-MSE", "qpu_yield_pct": 90.0, "throughput_pairs_per_s": 500.0,
         "qpu_cycles_saved_pct": 50.0, "energy_saved_pct": 40.0, "latency_ratio_c": 0.02},
    ]


def test_vertex_enumeration_covers_2_pow_n_combinations():
    summary_df, trials_df = run_weight_sensitivity_analysis(_dominant_rows(), BASE_WEIGHTS, perturbation_pct=0.10)
    assert len(trials_df) == 2 ** len(BASE_WEIGHTS)


def test_dominant_model_wins_every_vertex():
    summary_df, trials_df = run_weight_sensitivity_analysis(_dominant_rows(), BASE_WEIGHTS, perturbation_pct=0.10)

    top = summary_df.iloc[0]
    assert top["Model"] == "EdgeLSTM+CS-MSE"
    assert top["Win Rate (%)"] == pytest.approx(100.0)
    assert top["Wins (of 2^n vertices)"] == 2 ** len(BASE_WEIGHTS)
    # Every other model must have zero wins.
    assert (summary_df.iloc[1:]["Win Rate (%)"] == 0.0).all()


def test_summarize_robustness_reports_full_robustness():
    summary_df, _ = run_weight_sensitivity_analysis(_dominant_rows(), BASE_WEIGHTS, perturbation_pct=0.10)
    verdict = summarize_robustness(summary_df)
    assert "EdgeLSTM+CS-MSE" in verdict
    assert "100%" in verdict or "PROVABLY INDEPENDENT" in verdict


def test_ambiguous_case_is_detected_as_weight_sensitive():
    # Two criteria, equal base weights, each model dominant on one
    # criterion: genuinely ambiguous -- winner must flip across some
    # vertices of the +/-10% hypercube.
    rows = [
        {"Model": "A", "crit1": 100.0, "crit2": 0.0},
        {"Model": "B", "crit1": 0.0, "crit2": 100.0},
    ]
    weights = {"crit1": 0.5, "crit2": 0.5}
    summary_df, trials_df = run_weight_sensitivity_analysis(rows, weights, perturbation_pct=0.10)

    assert summary_df.iloc[0]["Win Rate (%)"] < 100.0
    assert set(trials_df["Winner"]) == {"A", "B"}

    verdict = summarize_robustness(summary_df)
    assert "sensitive" in verdict.lower()


def test_vertex_weights_are_within_perturbation_bounds():
    _, trials_df = run_weight_sensitivity_analysis(_dominant_rows(), BASE_WEIGHTS, perturbation_pct=0.10)
    for criterion, base in BASE_WEIGHTS.items():
        col = trials_df[f"w[{criterion}]"]
        assert (col >= base * 0.9 - 1e-12).all()
        assert (col <= base * 1.1 + 1e-12).all()


def test_all_sign_combinations_are_present_exactly_once():
    _, trials_df = run_weight_sensitivity_analysis(_dominant_rows(), BASE_WEIGHTS, perturbation_pct=0.10)
    signs_seen = set()
    for _, row in trials_df.iterrows():
        signs = tuple(
            1 if row[f"w[{c}]"] > BASE_WEIGHTS[c] else -1
            for c in BASE_WEIGHTS
        )
        signs_seen.add(signs)
    assert signs_seen == set(itertools.product([-1, 1], repeat=len(BASE_WEIGHTS)))


def test_empty_rows_raises():
    with pytest.raises(ValueError):
        run_weight_sensitivity_analysis([], BASE_WEIGHTS)


def test_invalid_perturbation_pct_raises():
    with pytest.raises(ValueError):
        run_weight_sensitivity_analysis(_dominant_rows(), BASE_WEIGHTS, perturbation_pct=1.0)
    with pytest.raises(ValueError):
        run_weight_sensitivity_analysis(_dominant_rows(), BASE_WEIGHTS, perturbation_pct=-0.1)


def test_summarize_robustness_empty_summary():
    import pandas as pd
    empty_df = pd.DataFrame(columns=["Model", "Win Rate (%)"])
    result = summarize_robustness(empty_df)
    assert "No models" in result
