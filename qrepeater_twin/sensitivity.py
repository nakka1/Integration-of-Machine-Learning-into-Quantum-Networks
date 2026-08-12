"""
Component 8 -- Decision-matrix weight sensitivity analysis
(`run_weight_sensitivity_analysis`).

Purpose: prove that the model ranked #1 by `metrics.build_decision_matrix`
(expected: `EdgeLSTM+CS-MSE`) wins REGARDLESS of the exact numeric weights
chosen for the decision criteria -- i.e. the conclusion is not an
artifact of having picked one particular, debatable set of weights.

Method and why it is exhaustive, not a spot-check
---------------------------------------------------
`build_decision_matrix` computes, for weights `w`, `Decision Score_m(w) =
sum_i (w_i / sum(w)) * norm_i,m` for each model `m`. Dividing by `sum(w)`
is a positive rescaling common to every model, so it can change the
*rank ordering* between two models `A`, `B` only if it flips the sign of

    Decision Score_A(w) - Decision Score_B(w) = (1 / sum(w)) * D_AB(w),
    where D_AB(w) = sum_i w_i * (norm_i,A - norm_i,B)  -- LINEAR in w.

`D_AB` is an affine (linear) function of the weight vector `w`. A linear
function over a bounded convex polytope attains BOTH its minimum and its
maximum at the polytope's vertices (standard linear-programming result).
Here the polytope is the hypercube `w_i in [(1 - pct) * base_i, (1 + pct)
* base_i]` for each criterion `i`, whose vertices are exactly the `2**n`
corner combinations of "-pct" / "+pct" per criterion. Consequently:

    Evaluating the decision matrix at all 2**n vertex combinations is
    SUFFICIENT -- not merely illustrative -- to certify whether the #1
    ranking changes anywhere inside the +/-pct box. If the same model
    wins at every vertex, `D_AB(w)` cannot change sign anywhere inside the
    box for any losing model `B` (its minimum over the box is already
    non-negative at the vertices), so that model wins EVERYWHERE inside
    the box, not just at the sampled points.

This is why the analysis below enumerates all `2**n` sign combinations
rather than randomly sampling perturbed weights.
"""

from __future__ import annotations

import itertools
from typing import Dict, List

import pandas as pd

from .metrics import build_decision_matrix


def run_weight_sensitivity_analysis(decision_rows: List[dict], base_weights: Dict[str, float],
                                     perturbation_pct: float = 0.10,
                                     model_key: str = "Model") -> tuple:
    """
    Combinatorial +/- `perturbation_pct` sensitivity analysis over the
    decision-matrix weights (see module docstring for why checking all
    `2**n` vertices is exhaustive, not a sample).

    Parameters
    ----------
    decision_rows : list[dict]
        Same input `build_decision_matrix` expects: one dict per candidate
        model, with `model_key` plus every criterion key present in
        `base_weights`.
    base_weights : dict[str, float]
        The nominal (unperturbed) decision weights, e.g.
        `ComparisonConfig.decision_weights`.
    perturbation_pct : float
        Fractional perturbation applied to each weight independently, in
        both directions (default 0.10, i.e. +/-10%).
    model_key : str
        Column identifying each row's model (default "Model").

    Returns
    -------
    summary_df : pd.DataFrame
        One row per candidate model: how many of the `2**n` vertex trials
        it won (`Wins`), the win rate (%), and its min/max Decision Score
        across all trials -- sorted by win rate, descending.
    trials_df : pd.DataFrame
        One row per vertex trial: the perturbed weight used for each
        criterion, the winning model, and its Decision Score.
    """
    if not decision_rows:
        raise ValueError("run_weight_sensitivity_analysis: decision_rows must be non-empty.")
    if not (0.0 <= perturbation_pct < 1.0):
        raise ValueError(f"run_weight_sensitivity_analysis: perturbation_pct must be in [0, 1), "
                          f"got {perturbation_pct!r}.")

    criteria = list(base_weights.keys())
    n = len(criteria)
    vertex_signs = list(itertools.product([-1, 1], repeat=n))

    trial_records = []
    winner_counts: Dict[str, int] = {}
    model_scores: Dict[str, list] = {}

    for signs in vertex_signs:
        perturbed_weights = {
            c: base_weights[c] * (1.0 + s * perturbation_pct)
            for c, s in zip(criteria, signs)
        }
        dm = build_decision_matrix(decision_rows, perturbed_weights, model_key=model_key)
        winner_row = dm.iloc[0]
        winner = winner_row[model_key]

        winner_counts[winner] = winner_counts.get(winner, 0) + 1
        trial_records.append({
            **{f"w[{c}]": perturbed_weights[c] for c in criteria},
            "Winner": winner,
            "Winner Decision Score": winner_row["Decision Score"],
        })
        for _, row in dm.iterrows():
            model_scores.setdefault(row[model_key], []).append(row["Decision Score"])

    n_trials = len(vertex_signs)
    summary_rows = []
    for model, scores in model_scores.items():
        wins = winner_counts.get(model, 0)
        summary_rows.append({
            model_key: model,
            "Wins (of 2^n vertices)": wins,
            "N Vertices": n_trials,
            "Win Rate (%)": wins / n_trials * 100.0,
            "Min Decision Score": min(scores),
            "Max Decision Score": max(scores),
        })

    summary_df = pd.DataFrame(summary_rows).sort_values(
        "Win Rate (%)", ascending=False
    ).reset_index(drop=True)
    trials_df = pd.DataFrame(trial_records)
    return summary_df, trials_df


def summarize_robustness(summary_df: pd.DataFrame, model_key: str = "Model") -> str:
    """
    Human-readable one-line verdict from `run_weight_sensitivity_analysis`'s
    `summary_df`: names the top model and states whether it won 100% of
    the `2**n` vertex trials (fully robust to +/-`perturbation_pct` weight
    bias) or only some fraction of them (rank is weight-dependent).
    """
    if summary_df.empty:
        return "No models to summarize."

    top = summary_df.iloc[0]
    model, win_rate = top[model_key], top["Win Rate (%)"]
    if win_rate >= 100.0 - 1e-9:
        return (f"'{model}' wins Rank #1 in 100% of the vertex trials -- the ranking is "
                f"PROVABLY INDEPENDENT of the exact decision-matrix weights within the "
                f"tested +/- range (see module docstring for why vertex enumeration is "
                f"exhaustive, not a sample).")
    return (f"'{model}' only wins Rank #1 in {win_rate:.1f}% of the vertex trials -- the "
            f"ranking IS sensitive to the exact decision-matrix weights within the tested "
            f"+/- range; inspect `trials_df` to see which criteria drive the flips.")
