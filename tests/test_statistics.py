import warnings

import pytest

from quantum_twin.statistics_tests import (
    compare_models_statistically,
    compute_confidence_interval,
    holm_bonferroni_correction,
    paired_ttest,
    wilcoxon_test,
)


# ---------------------------------------------------------------------------
# compute_confidence_interval
# ---------------------------------------------------------------------------

def test_confidence_interval_contains_mean():
    values = [10, 12, 11, 13, 9]
    mean, lo, hi = compute_confidence_interval(values)
    assert mean == pytest.approx(11.0)
    assert lo < mean < hi


def test_confidence_interval_narrows_with_more_samples():
    low_n = [10, 12, 11, 13, 9]
    high_n = low_n * 4  # same mean/std, 4x the sample size
    _, lo1, hi1 = compute_confidence_interval(low_n)
    _, lo2, hi2 = compute_confidence_interval(high_n)
    assert (hi2 - lo2) < (hi1 - lo1)


def test_confidence_interval_degenerate_single_sample():
    mean, lo, hi = compute_confidence_interval([5.0])
    assert mean == 5.0
    assert lo != lo and hi != hi  # NaN


def test_confidence_interval_empty():
    mean, lo, hi = compute_confidence_interval([])
    assert mean != mean


def test_confidence_interval_zero_variance():
    mean, lo, hi = compute_confidence_interval([7.0, 7.0, 7.0])
    assert mean == lo == hi == 7.0


# ---------------------------------------------------------------------------
# paired_ttest / wilcoxon_test
# ---------------------------------------------------------------------------

def test_paired_ttest_detects_clear_difference():
    a = [10, 11, 12, 13, 14]
    b = [5, 6, 7, 8, 9]
    t_stat, p = paired_ttest(a, b)
    assert p < 0.001
    assert t_stat > 0  # a consistently greater than b


def test_paired_ttest_requires_matching_length():
    with pytest.raises(ValueError):
        paired_ttest([1, 2, 3], [1, 2])


def test_paired_ttest_requires_at_least_two_observations():
    with pytest.raises(ValueError):
        paired_ttest([1], [2])


def test_wilcoxon_test_detects_clear_difference():
    a = [10, 11, 12, 13, 14]
    b = [5, 6, 7, 8, 9]
    _, p = wilcoxon_test(a, b)
    assert p < 0.1


def test_wilcoxon_test_identical_arrays_does_not_crash():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        stat, p = wilcoxon_test([10] * 5, [10] * 5)
    # Either scipy raises internally (caught -> nan) or returns p=1.0;
    # either way this must not propagate an exception to the caller.
    assert (p != p) or (p == 1.0)


# ---------------------------------------------------------------------------
# holm_bonferroni_correction
# ---------------------------------------------------------------------------

def test_holm_bonferroni_matches_manual_step_down():
    pvals = [0.01, 0.04, 0.03, 0.20, 0.001]
    adjusted, reject = holm_bonferroni_correction(pvals, alpha=0.05)
    expected = [0.04, 0.09, 0.09, 0.20, 0.005]
    for e, a in zip(expected, adjusted):
        assert a == pytest.approx(e, abs=1e-9)
    assert reject == [True, False, False, False, True]


def test_holm_bonferroni_adjusted_pvalues_are_monotonic_in_sorted_order():
    pvals = [0.5, 0.001, 0.3, 0.02, 0.15]
    adjusted, _ = holm_bonferroni_correction(pvals)
    order = sorted(range(len(pvals)), key=lambda i: pvals[i])
    sorted_adjusted = [adjusted[i] for i in order]
    assert sorted_adjusted == sorted(sorted_adjusted)


def test_holm_bonferroni_never_exceeds_one():
    pvals = [0.9, 0.8, 0.99]
    adjusted, _ = holm_bonferroni_correction(pvals)
    assert all(a <= 1.0 for a in adjusted)


def test_holm_bonferroni_handles_nan_entries():
    pvals = [0.01, float("nan"), 0.03]
    adjusted, reject = holm_bonferroni_correction(pvals, alpha=0.05)
    assert adjusted[1] != adjusted[1]  # still NaN
    assert reject[1] is False
    assert reject[0] is True  # unaffected by the NaN entry


def test_holm_bonferroni_all_nan_returns_all_false():
    adjusted, reject = holm_bonferroni_correction([float("nan"), float("nan")])
    assert all(r is False for r in reject)


def test_holm_bonferroni_empty_input():
    adjusted, reject = holm_bonferroni_correction([])
    assert adjusted == [] and reject == []


def test_holm_bonferroni_single_pvalue_equals_itself():
    adjusted, reject = holm_bonferroni_correction([0.03], alpha=0.05)
    assert adjusted[0] == pytest.approx(0.03)
    assert reject[0] is True


# ---------------------------------------------------------------------------
# compare_models_statistically
# ---------------------------------------------------------------------------

def _make_seed_results(seeds, values):
    return [{"seed": s, "qpu_yield_pct": v} for s, v in zip(seeds, values)]


def test_compare_models_statistically_flags_clear_improvement():
    per_model_seed_results = {
        "EdgeLSTM+CS-MSE": _make_seed_results([42, 43, 44, 45, 46], [85, 86, 87, 84, 88]),
        "LSTM+MSE": _make_seed_results([42, 43, 44, 45, 46], [60, 61, 59, 62, 60]),
    }
    df = compare_models_statistically(per_model_seed_results, "qpu_yield_pct", reference_model="EdgeLSTM+CS-MSE")
    assert len(df) == 1
    row = df.iloc[0]
    assert row["N Paired Seeds"] == 5
    assert row["Mean Difference"] < 0  # LSTM+MSE scores lower than EdgeLSTM+CS-MSE
    assert row["p-value (t-test)"] < 0.01
    assert bool(row[[c for c in df.columns if c.startswith("Significant (t-test, Holm")][0]])


def test_compare_models_statistically_handles_single_seed_model():
    per_model_seed_results = {
        "EdgeLSTM+CS-MSE": _make_seed_results([42, 43, 44, 45, 46], [85, 86, 87, 84, 88]),
        "Oracle": _make_seed_results([42], [99.0]),
    }
    df = compare_models_statistically(per_model_seed_results, "qpu_yield_pct", reference_model="EdgeLSTM+CS-MSE")
    row = df[df["Model"] == "Oracle"].iloc[0]
    assert row["N Paired Seeds"] == 1
    # Below min_n_for_tests: p-values must be NaN, not silently computed on n=1.
    assert row["p-value (t-test)"] != row["p-value (t-test)"]


def test_compare_models_statistically_uses_only_common_seeds():
    per_model_seed_results = {
        "EdgeLSTM+CS-MSE": _make_seed_results([42, 43, 44], [85, 86, 87]),
        "LSTM+MSE": _make_seed_results([43, 44, 45], [60, 61, 62]),  # seed 45 not shared with reference
    }
    df = compare_models_statistically(per_model_seed_results, "qpu_yield_pct", reference_model="EdgeLSTM+CS-MSE")
    row = df.iloc[0]
    assert row["N Paired Seeds"] == 2  # only seeds 43, 44 are common


def test_compare_models_statistically_default_comparison_models_excludes_reference():
    per_model_seed_results = {
        "A": _make_seed_results([1, 2, 3], [1, 2, 3]),
        "B": _make_seed_results([1, 2, 3], [4, 5, 6]),
        "C": _make_seed_results([1, 2, 3], [7, 8, 9]),
    }
    df = compare_models_statistically(per_model_seed_results, "qpu_yield_pct", reference_model="A")
    assert set(df["Model"]) == {"B", "C"}


def test_compare_models_statistically_holm_correction_applied_across_family():
    # Three comparisons with progressively weaker evidence; Holm correction
    # should make the family-wise significance flags stricter than raw p<0.05.
    per_model_seed_results = {
        "Ref": _make_seed_results(range(10), [10 + i * 0.01 for i in range(10)]),
        "Close1": _make_seed_results(range(10), [10.5 + i * 0.01 for i in range(10)]),
        "Close2": _make_seed_results(range(10), [10.6 + i * 0.01 for i in range(10)]),
        "Close3": _make_seed_results(range(10), [10.7 + i * 0.01 for i in range(10)]),
    }
    df = compare_models_statistically(per_model_seed_results, "qpu_yield_pct", reference_model="Ref")
    holm_col = [c for c in df.columns if c.startswith("p-value (t-test, Holm")][0]
    raw_col = "p-value (t-test)"
    # Holm-adjusted p-values must never be smaller than the raw p-values.
    assert (df[holm_col] >= df[raw_col] - 1e-9).all()
