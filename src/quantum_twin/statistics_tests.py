"""
Component 12 -- Statistical significance testing.

Every comparison table elsewhere in this project (`results_df` in
`pareto_sweep.py` / `model_comparison.py` / `ablation.py`) reports only
mean +/- standard deviation across seeds. That is descriptive, not
inferential: it does not say whether an observed difference between two
models is larger than what seed-to-seed noise alone would produce, and
comparing many models/metrics at once inflates the chance that SOME
difference looks "real" by pure chance (the multiple-comparisons
problem).

This module adds the inferential layer:

    - `compute_confidence_interval` : a t-distribution-based 95% CI for a
      small sample (n=5-15 seeds), rather than only a point estimate.
    - `paired_ttest` / `wilcoxon_test` : two complementary paired tests
      for "is model A different from model B on this metric, seed by
      seed?" -- the paired t-test assumes approximately normal
      differences; the Wilcoxon signed-rank test is its non-parametric,
      assumption-light counterpart, included precisely because n is small
      enough that normality is not a safe default assumption.
    - `holm_bonferroni_correction` : controls the family-wise error rate
      across a whole family of comparisons (e.g. EdgeLSTM+CS-MSE vs. every
      other model, for one metric) using the Holm step-down procedure --
      uniformly more powerful than a flat Bonferroni correction while
      giving the same family-wise error guarantee.
    - `compare_models_statistically` : the high-level entry point tying
      all of the above together into one results table, given the
      `per_model_seed_results` / `per_cell_seed_results` /
      `per_seed_results` dict already produced by
      `model_comparison.run_model_comparison` / `ablation.run_ablation_study`
      / `pareto_sweep.run_pareto_sweep`.
"""

from __future__ import annotations

from typing import Dict, List, Sequence

import numpy as np
import pandas as pd
from scipy import stats


# ---------------------------------------------------------------------------
# Confidence intervals
# ---------------------------------------------------------------------------

def compute_confidence_interval(values: Sequence[float], confidence: float = 0.95) -> tuple:
    """
    Two-sided `confidence`-level confidence interval for the mean of
    `values`, using the t-distribution (appropriate for the small sample
    sizes -- typically n=5 to n=15 seeds -- used throughout this project,
    where the normal-approximation/z-interval would understate the true
    uncertainty).

    Returns `(mean, ci_low, ci_high)`. With `n < 2` samples, no interval
    is defined; returns `(mean, nan, nan)` rather than raising, so callers
    aggregating many metrics don't need a special case for degenerate
    single-fold/single-seed inputs.
    """
    arr = np.asarray(list(values), dtype=float)
    n = len(arr)
    mean = float(np.mean(arr)) if n > 0 else float("nan")
    if n < 2:
        return mean, float("nan"), float("nan")

    sem = stats.sem(arr)  # standard error of the mean
    if sem == 0:
        return mean, mean, mean
    half_width = sem * stats.t.ppf((1.0 + confidence) / 2.0, df=n - 1)
    return mean, mean - half_width, mean + half_width


# ---------------------------------------------------------------------------
# Paired significance tests
# ---------------------------------------------------------------------------

def paired_ttest(a: Sequence[float], b: Sequence[float]) -> tuple:
    """
    Paired (dependent-samples) t-test: `H0: mean(a - b) == 0`. `a` and `b`
    must be seed-aligned (same length, `a[i]`/`b[i]` from the SAME seed) --
    see `compare_models_statistically`, which handles the alignment.

    Assumes the per-seed DIFFERENCES `a - b` are approximately normally
    distributed; with very few seeds this assumption is hard to verify, so
    `wilcoxon_test` is provided alongside as an assumption-light
    alternative and both are reported by `compare_models_statistically`.

    Returns `(statistic, p_value)`. Raises `ValueError` if `len(a) !=
    len(b)` or `len(a) < 2`.
    """
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    if len(a) != len(b):
        raise ValueError(f"paired_ttest: a and b must be the same length ({len(a)} vs {len(b)}).")
    if len(a) < 2:
        raise ValueError("paired_ttest: need at least 2 paired observations.")
    result = stats.ttest_rel(a, b)
    return float(result.statistic), float(result.pvalue)


def wilcoxon_test(a: Sequence[float], b: Sequence[float]) -> tuple:
    """
    Wilcoxon signed-rank test: the non-parametric counterpart to
    `paired_ttest`, testing whether the paired differences `a - b` are
    symmetric around zero, WITHOUT assuming they are normally
    distributed -- the safer default when only a handful of seeds are
    available.

    Returns `(statistic, p_value)`, or `(nan, nan)` if the test is not
    computable (e.g. fewer than 2 non-zero paired differences remain after
    scipy's default zero-handling). Behavior for the degenerate
    all-identical-pairs case varies across `scipy` versions -- some raise
    `ValueError`, others return `(0.0, 1.0)` with a `RuntimeWarning` about
    a zero standard error; either way, this function never crashes the
    caller: the `ValueError` case is caught and mapped to `(nan, nan)`,
    and the warning-but-succeeds case is returned as-is (a p-value of 1.0
    correctly reflects "no evidence of any difference").
    """
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    if len(a) != len(b):
        raise ValueError(f"wilcoxon_test: a and b must be the same length ({len(a)} vs {len(b)}).")
    if len(a) < 2:
        raise ValueError("wilcoxon_test: need at least 2 paired observations.")
    try:
        result = stats.wilcoxon(a, b)
        return float(result.statistic), float(result.pvalue)
    except ValueError:
        # All differences are zero, or too few non-zero differences remain.
        return float("nan"), float("nan")


# ---------------------------------------------------------------------------
# Multiple-comparisons correction
# ---------------------------------------------------------------------------

def holm_bonferroni_correction(p_values: Sequence[float], alpha: float = 0.05) -> tuple:
    """
    Holm-Bonferroni step-down procedure: controls the family-wise error
    rate across `len(p_values)` simultaneous hypothesis tests (e.g.
    "EdgeLSTM+CS-MSE vs. every other model", one p-value per comparison)
    at level `alpha`, while being uniformly more powerful than a flat
    Bonferroni correction (`p * m`) for the same guarantee.

    Algorithm: sort p-values ascending; the k-th smallest (1-indexed) is
    compared against `alpha / (m - k + 1)`; reject H0 for it and every
    smaller p-value up to (but not including) the first failure to reject.
    Adjusted p-values are computed as the standard step-down running
    maximum of `p_(k) * (m - k + 1)`, capped at 1.0, so that
    `adjusted_p[i] <= alpha` is equivalent to "reject H0 for comparison
    i" -- the conventional way to report Holm-adjusted results.

    `nan` entries in `p_values` (e.g. from `wilcoxon_test`'s degenerate
    case) are passed through as `nan` in the output and excluded from the
    correction family (a family of size `m` = the count of non-nan
    p-values).

    Returns `(adjusted_p_values, reject)`, both lists the same length and
    order as `p_values`; `reject[i]` is `True` iff `adjusted_p_values[i] <=
    alpha`.
    """
    p_values = list(p_values)
    m_total = len(p_values)
    valid_idx = [i for i, p in enumerate(p_values) if p == p]  # drop NaNs
    m = len(valid_idx)

    adjusted = [float("nan")] * m_total
    if m == 0:
        return adjusted, [False] * m_total

    order = sorted(valid_idx, key=lambda i: p_values[i])
    running_max = 0.0
    for rank, idx in enumerate(order):  # rank is 0-indexed; factor = m - rank
        factor = m - rank
        adj = min(1.0, p_values[idx] * factor)
        running_max = max(running_max, adj)  # enforce monotonicity (standard step-down requirement)
        adjusted[idx] = running_max

    reject = [(adjusted[i] == adjusted[i]) and (adjusted[i] <= alpha) for i in range(m_total)]
    return adjusted, reject


# ---------------------------------------------------------------------------
# High-level entry point
# ---------------------------------------------------------------------------

def _aligned_pair(per_model_seed_results: Dict[str, List[dict]], reference_model: str,
                   comparison_model: str, metric_key: str) -> tuple:
    """
    Extracts `(reference_values, comparison_values)`, seed-aligned:
    only seeds present in BOTH models' per-seed results are kept (in
    matching order), since `paired_ttest`/`wilcoxon_test` require a strict
    1-to-1 pairing. This matters in particular for models evaluated on a
    single nominal seed (e.g. `baselines.Persistence`/`MovingAverage`/
    `Oracle` in `model_comparison.run_model_comparison`, which are
    deterministic and therefore only run once) -- comparisons involving
    them naturally reduce to n=1 and are flagged as such by
    `compare_models_statistically` rather than silently misaligned.
    """
    ref_by_seed = {r["seed"]: r[metric_key] for r in per_model_seed_results[reference_model]}
    cmp_by_seed = {r["seed"]: r[metric_key] for r in per_model_seed_results[comparison_model]}
    common_seeds = sorted(set(ref_by_seed) & set(cmp_by_seed))
    ref_values = [ref_by_seed[s] for s in common_seeds]
    cmp_values = [cmp_by_seed[s] for s in common_seeds]
    return ref_values, cmp_values


def compare_models_statistically(per_model_seed_results: Dict[str, List[dict]], metric_key: str,
                                  reference_model: str, comparison_models: List[str] | None = None,
                                  alpha: float = 0.05, confidence: float = 0.95,
                                  min_n_for_tests: int = 4) -> pd.DataFrame:
    """
    Statistically compares `reference_model` (typically
    `"EdgeLSTM+CS-MSE"`) against one or more other models on a single
    `metric_key` (a key present in each per-seed result dict, e.g.
    `"qpu_yield_pct"`, `"mae"`), seed by seed.

    Parameters
    ----------
    per_model_seed_results : dict[str, list[dict]]
        As returned by `model_comparison.run_model_comparison` /
        `ablation.run_ablation_study` (there keyed by `(architecture,
        loss)` tuples rather than plain names -- convert keys to strings
        first if comparing ablation cells) / `pareto_sweep.run_pareto_sweep`.
    metric_key : str
        The per-seed dict key to compare (e.g. `"qpu_yield_pct"`, `"mae"`,
        `"fp"`).
    reference_model : str
        The model every other model in `comparison_models` is compared
        AGAINST (mean difference = comparison - reference; positive means
        the comparison model scores higher on `metric_key`).
    comparison_models : list[str], optional
        Which models to compare against `reference_model`. Default: every
        other key in `per_model_seed_results`.
    min_n_for_tests : int
        Below this many paired (seed-aligned) observations, the paired
        t-test / Wilcoxon test are NOT run (too few points for either
        test's assumptions to be remotely reasonable) and their p-value
        columns are left as `nan` -- `Mean Difference` and the CI are
        still reported down to n=2, since a point estimate with a wide CI
        is still informative, unlike a p-value computed on 2-3 points.

    Returns
    -------
    pd.DataFrame, one row per compared model, columns:
        Model, Metric, N Paired Seeds,
        Mean Reference, Mean Comparison, Mean Difference,
        95% CI Low, 95% CI High (of the paired difference),
        t-statistic, p-value (t-test), p-value (t-test, Holm),
        Significant (t-test, Holm, alpha),
        W-statistic, p-value (Wilcoxon), p-value (Wilcoxon, Holm),
        Significant (Wilcoxon, Holm, alpha)

    The Holm-Bonferroni correction is applied ACROSS all rows of this
    table (i.e. across every `comparison_models` entry for this one
    `metric_key` call) -- one correction "family" per call, matching the
    common reporting pattern "compare the proposed model against every
    baseline, correct for the resulting multiple comparisons".
    """
    comparison_models = comparison_models or [m for m in per_model_seed_results if m != reference_model]

    rows = []
    raw_t_pvalues, raw_w_pvalues = [], []

    for model in comparison_models:
        ref_values, cmp_values = _aligned_pair(per_model_seed_results, reference_model, model, metric_key)
        n = len(ref_values)
        diffs = [c - r for r, c in zip(ref_values, cmp_values)]

        mean_diff, ci_low, ci_high = compute_confidence_interval(diffs, confidence=confidence)
        mean_ref = float(np.mean(ref_values)) if ref_values else float("nan")
        mean_cmp = float(np.mean(cmp_values)) if cmp_values else float("nan")

        t_stat = t_p = w_stat = w_p = float("nan")
        if n >= min_n_for_tests:
            t_stat, t_p = paired_ttest(cmp_values, ref_values)
            w_stat, w_p = wilcoxon_test(cmp_values, ref_values)

        rows.append({
            "Model": model, "Metric": metric_key, "N Paired Seeds": n,
            "Mean Reference": mean_ref, "Mean Comparison": mean_cmp, "Mean Difference": mean_diff,
            "95% CI Low": ci_low, "95% CI High": ci_high,
            "t-statistic": t_stat, "p-value (t-test)": t_p,
            "W-statistic": w_stat, "p-value (Wilcoxon)": w_p,
        })
        raw_t_pvalues.append(t_p)
        raw_w_pvalues.append(w_p)

    t_adjusted, t_reject = holm_bonferroni_correction(raw_t_pvalues, alpha=alpha)
    w_adjusted, w_reject = holm_bonferroni_correction(raw_w_pvalues, alpha=alpha)

    for row, t_adj, t_rej, w_adj, w_rej in zip(rows, t_adjusted, t_reject, w_adjusted, w_reject):
        row["p-value (t-test, Holm)"] = t_adj
        row[f"Significant (t-test, Holm, alpha={alpha})"] = t_rej
        row["p-value (Wilcoxon, Holm)"] = w_adj
        row[f"Significant (Wilcoxon, Holm, alpha={alpha})"] = w_rej

    return pd.DataFrame(rows, columns=[
        "Model", "Metric", "N Paired Seeds", "Mean Reference", "Mean Comparison", "Mean Difference",
        "95% CI Low", "95% CI High", "t-statistic", "p-value (t-test)", "p-value (t-test, Holm)",
        f"Significant (t-test, Holm, alpha={alpha})", "W-statistic", "p-value (Wilcoxon)",
        "p-value (Wilcoxon, Holm)", f"Significant (Wilcoxon, Holm, alpha={alpha})",
    ])
