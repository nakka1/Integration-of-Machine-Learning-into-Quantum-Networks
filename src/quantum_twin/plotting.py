"""
Component 10 -- Plotting helpers.

Every function here takes the plain `pd.DataFrame`/`dict` outputs already
produced by `pareto_sweep.py`, `model_comparison.py`, `ablation.py`,
`sensitivity.py`, and `metrics.prediction`, and returns a
`matplotlib.figure.Figure` -- never calls `plt.show()` itself, so the
caller (notebook cell or `experiment_tracking.ExperimentRun`) decides
whether to display it inline, save it to disk, or both.

Each function accepts an optional `save_path`: when given, the figure is
also written to disk (`fig.savefig(save_path, dpi=150,
bbox_inches="tight")`) before being returned.

`_parse_mean_std_column` is the shared adapter that turns this project's
display-formatted `"12.34 +/- 1.02"` string columns (see `pareto_sweep.py`
/ `model_comparison.py` / `ablation.py`) back into numeric
`(means, stds)` arrays for plotting with error bars, so callers never need
to re-run an experiment just to get numbers instead of strings.
"""

from __future__ import annotations

import re
from typing import Sequence

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from matplotlib.figure import Figure

_MEAN_STD_RE = re.compile(r"^\s*([+-]?[\d.eE+-]+)\s*\+/-\s*([\d.eE+-]+)\s*$")


def _parse_mean_std_column(series: pd.Series) -> tuple:
    """
    Parses a `"<mean> +/- <std>"` string column (as produced throughout
    this project's `results_df` tables) into two numpy float arrays
    `(means, stds)`. Raises `ValueError` on the first entry that doesn't
    match the expected format.
    """
    means, stds = [], []
    for value in series:
        m = _MEAN_STD_RE.match(str(value))
        if not m:
            raise ValueError(f"_parse_mean_std_column: could not parse {value!r} as '<mean> +/- <std>'.")
        means.append(float(m.group(1)))
        stds.append(float(m.group(2)))
    return np.array(means), np.array(stds)


def _finish(fig: Figure, save_path: str | None = None) -> Figure:
    fig.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


# ---------------------------------------------------------------------------
# Pareto sweep
# ---------------------------------------------------------------------------

def plot_pareto_frontier(results_df: pd.DataFrame, x_col: str = "Lambda",
                          metric_cols: Sequence[str] = ("QPU Yield (%)", "MAE"),
                          save_path: str | None = None) -> Figure:
    """
    Line plot of one or more mean +/- std metric columns against
    `x_col` (default `lambda_penalty`), one subplot per metric, sharing
    the x-axis. Designed for `pareto_sweep.run_pareto_sweep`'s
    `results_df`.
    """
    x = results_df[x_col].astype(float).to_numpy()
    n = len(metric_cols)
    fig, axes = plt.subplots(n, 1, figsize=(7, 3.2 * n), sharex=True)
    if n == 1:
        axes = [axes]

    for ax, col in zip(axes, metric_cols):
        means, stds = _parse_mean_std_column(results_df[col])
        ax.errorbar(x, means, yerr=stds, marker="o", capsize=3)
        ax.set_ylabel(col)
        ax.grid(alpha=0.3)

    axes[-1].set_xlabel(x_col)
    fig.suptitle("Pareto Frontier: mean +/- std across seeds")
    return _finish(fig, save_path)


# ---------------------------------------------------------------------------
# Model comparison / ablation bar charts
# ---------------------------------------------------------------------------

def plot_model_comparison_bars(results_df: pd.DataFrame, metric_col: str = "QPU Yield (%)",
                                model_col: str = "Model", save_path: str | None = None) -> Figure:
    """
    Bar chart of `metric_col` (a `"<mean> +/- <std>"` column) across
    models/grid cells, with error bars. Works for both
    `model_comparison.run_model_comparison`'s and
    `ablation.run_ablation_study`'s `results_df`.
    """
    means, stds = _parse_mean_std_column(results_df[metric_col])
    labels = results_df[model_col].astype(str).to_numpy()

    fig, ax = plt.subplots(figsize=(max(6, 1.1 * len(labels)), 4.5))
    x = np.arange(len(labels))
    ax.bar(x, means, yerr=stds, capsize=4)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_ylabel(metric_col)
    ax.set_title(f"{metric_col} by model (mean +/- std across seeds)")
    ax.grid(alpha=0.3, axis="y")
    return _finish(fig, save_path)


def plot_ablation_interaction(decomposition_df: pd.DataFrame, metric: str, save_path: str | None = None) -> Figure:
    """
    Classic 2x2 factorial interaction plot for one metric: x-axis = loss
    function (MSE, CS-MSE), one line per architecture (EdgeLSTM,
    StandardLSTM). Parallel lines indicate a purely additive
    (non-interacting) effect; non-parallel lines make the interaction
    effect visually explicit -- the same quantity reported numerically
    in `decomposition_df["Interaction Effect"]`.
    """
    row = decomposition_df[decomposition_df["Metric"] == metric]
    if row.empty:
        raise ValueError(f"plot_ablation_interaction: metric {metric!r} not found in decomposition_df.")
    row = row.iloc[0]

    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    x = [0, 1]  # MSE, CS-MSE
    ax.plot(x, [row["EdgeLSTM+MSE"], row["EdgeLSTM+CS-MSE"]], marker="o", label="EdgeLSTM")
    ax.plot(x, [row["StandardLSTM+MSE"], row["StandardLSTM+CS-MSE"]], marker="o", label="StandardLSTM")
    ax.set_xticks(x)
    ax.set_xticklabels(["MSE", "CS-MSE"])
    ax.set_ylabel(metric)
    ax.set_title(f"Architecture x Loss interaction: {metric}")
    ax.legend()
    ax.grid(alpha=0.3)
    return _finish(fig, save_path)


# ---------------------------------------------------------------------------
# Confusion matrix / decision metrics
# ---------------------------------------------------------------------------

def plot_confusion_matrix(confusion: dict[str, int], title: str = "Admission Confusion Matrix",
                           save_path: str | None = None) -> Figure:
    """
    2x2 heatmap of the admission confusion matrix
    `{"TP", "FP", "TN", "FN"}` (see `orchestrator.py` / `metrics.decision`
    for the ground-truth/predicted-label convention).
    """
    matrix = np.array([[confusion["TP"], confusion["FN"]],
                        [confusion["FP"], confusion["TN"]]])
    fig, ax = plt.subplots(figsize=(4.5, 4))
    im = ax.imshow(matrix, cmap="Blues")
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Predicted Admit", "Predicted Halt"])
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["True Good", "True Bad"])
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(matrix[i, j]), ha="center", va="center",
                     color="white" if matrix[i, j] > matrix.max() / 2 else "black")
    ax.set_title(title)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    return _finish(fig, save_path)


def plot_decision_matrix(decision_matrix_df: pd.DataFrame, model_col: str = "Model",
                          save_path: str | None = None) -> Figure:
    """
    Horizontal bar chart of `Decision Score` by model, ordered by `Rank`
    (rank 1 at the top). Designed for `metrics.decision.build_decision_matrix`'s
    output.
    """
    df = decision_matrix_df.sort_values("Rank")
    fig, ax = plt.subplots(figsize=(6.5, max(3, 0.6 * len(df))))
    y = np.arange(len(df))
    ax.barh(y, df["Decision Score"].to_numpy())
    ax.set_yticks(y)
    ax.set_yticklabels(df[model_col].astype(str).to_numpy())
    ax.invert_yaxis()  # rank 1 on top
    ax.set_xlabel("Decision Score (higher = better)")
    ax.set_title("Multi-criteria decision matrix ranking")
    ax.grid(alpha=0.3, axis="x")
    return _finish(fig, save_path)


# ---------------------------------------------------------------------------
# Sensitivity analysis
# ---------------------------------------------------------------------------

def plot_sensitivity_summary(summary_df: pd.DataFrame, model_col: str = "Model",
                              save_path: str | None = None) -> Figure:
    """
    Bar chart of `Win Rate (%)` per model from
    `sensitivity.run_weight_sensitivity_analysis`'s `summary_df` -- how
    often each model ranks #1 across the `2**n` +/-pct weight-perturbation
    vertices.
    """
    df = summary_df.sort_values("Win Rate (%)", ascending=False)
    fig, ax = plt.subplots(figsize=(max(6, 1.1 * len(df)), 4.5))
    x = np.arange(len(df))
    ax.bar(x, df["Win Rate (%)"].to_numpy())
    ax.set_xticks(x)
    ax.set_xticklabels(df[model_col].astype(str).to_numpy(), rotation=30, ha="right")
    ax.set_ylabel("Win Rate (%) across weight-perturbation vertices")
    ax.set_ylim(0, 105)
    ax.set_title("Decision-weight sensitivity: Rank #1 win rate")
    ax.grid(alpha=0.3, axis="y")
    return _finish(fig, save_path)


# ---------------------------------------------------------------------------
# Temporal prediction analysis
# ---------------------------------------------------------------------------

def plot_temporal_prediction_error(temporal_metrics: dict, save_path: str | None = None) -> Figure:
    """
    Histogram of matched threshold-crossing timing errors (in steps) from
    `metrics.prediction.compute_temporal_prediction_metrics` /
    `compute_controller_decision_timing`'s output. A vertical line at 0
    marks perfect timing; bars to the right of 0 are LATE detections
    (positive `timing_error_steps`), bars to the left are EARLY /
    anticipatory ones.
    """
    errors = temporal_metrics.get("timing_errors_steps", [])
    fig, ax = plt.subplots(figsize=(6, 4))
    if errors:
        ax.hist(errors, bins=min(20, max(5, len(errors))), color="tab:blue", alpha=0.8)
    ax.axvline(0.0, color="black", linestyle="--", linewidth=1)
    ax.set_xlabel("Timing error (steps); + = late/lag, - = early/anticipation")
    ax.set_ylabel("Count of matched degradation events")
    n_missed = temporal_metrics.get("n_missed_events", 0)
    n_false = temporal_metrics.get("n_false_alarms", 0)
    ax.set_title(f"Threshold-crossing timing error (missed={n_missed}, false alarms={n_false})")
    ax.grid(alpha=0.3)
    return _finish(fig, save_path)


def plot_fidelity_timeseries_with_crossings(
    y_true: np.ndarray, y_pred: np.ndarray, threshold: float,
    true_crossings: np.ndarray | None = None, pred_crossings: np.ndarray | None = None,
    save_path: str | None = None,
) -> Figure:
    """
    Line plot of the true vs. predicted fidelity trajectory, the
    admission threshold, and (optionally) marked degradation-event
    crossings for both series -- a diagnostic figure pairing directly
    with `metrics.prediction.compute_temporal_prediction_metrics`.
    """
    y_true = np.asarray(y_true, dtype=float).ravel()
    y_pred = np.asarray(y_pred, dtype=float).ravel()
    t = np.arange(len(y_true))

    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.plot(t, y_true, label="True fidelity F(t)", color="tab:blue")
    ax.plot(t, y_pred, label="Predicted fidelity F_hat(t)", color="tab:orange", alpha=0.85)
    ax.axhline(threshold, color="black", linestyle="--", linewidth=1, label=f"Threshold = {threshold}")

    if true_crossings is not None and len(true_crossings) > 0:
        ax.scatter(true_crossings, [threshold] * len(true_crossings),
                   marker="v", color="tab:blue", s=60, zorder=5, label="True degradation events")
    if pred_crossings is not None and len(pred_crossings) > 0:
        ax.scatter(pred_crossings, [threshold] * len(pred_crossings),
                   marker="^", color="tab:orange", s=60, zorder=5, label="Predicted degradation events")

    ax.set_xlabel("Time step")
    ax.set_ylabel("Fidelity")
    ax.set_title("Fidelity trajectory: true vs. predicted, with degradation crossings")
    ax.legend(loc="best", fontsize=8)
    ax.grid(alpha=0.3)
    return _finish(fig, save_path)


# ---------------------------------------------------------------------------
# Statistical significance (forest plot)
# ---------------------------------------------------------------------------

def plot_significance_forest(significance_df: pd.DataFrame, save_path: str | None = None) -> Figure:
    """
    Forest plot of `statistics_tests.compare_models_statistically`'s
    output: one horizontal error bar per compared model, centered on
    `"Mean Difference"` (comparison - reference) with whiskers at
    `"95% CI Low"`/`"95% CI High"`, and a vertical dashed line at 0 (no
    effect). Markers whose Holm-adjusted t-test p-value is significant
    are filled solid; non-significant ones are hollow -- the standard,
    at-a-glance way to present a family of paired statistical comparisons
    with both effect size AND significance visible in one figure.
    """
    df = significance_df.reset_index(drop=True)
    sig_col = next((c for c in df.columns if c.startswith("Significant (t-test, Holm")), None)

    fig, ax = plt.subplots(figsize=(7, max(3, 0.6 * len(df))))
    y = np.arange(len(df))

    means = df["Mean Difference"].to_numpy()
    lo = df["95% CI Low"].to_numpy()
    hi = df["95% CI High"].to_numpy()
    xerr = np.vstack([means - lo, hi - means])

    significant = df[sig_col].to_numpy() if sig_col is not None else np.zeros(len(df), dtype=bool)

    for i in range(len(df)):
        color = "tab:red" if significant[i] else "tab:gray"
        marker = "o" if significant[i] else "o"
        facecolor = color if significant[i] else "white"
        ax.errorbar(means[i], y[i], xerr=[[xerr[0, i]], [xerr[1, i]]], fmt=marker,
                     color=color, markerfacecolor=facecolor, markeredgecolor=color,
                     capsize=4, markersize=7)

    ax.axvline(0.0, color="black", linestyle="--", linewidth=1)
    ax.set_yticks(y)
    ax.set_yticklabels(df["Model"].astype(str).to_numpy())
    ax.invert_yaxis()
    metric = df["Metric"].iloc[0] if "Metric" in df.columns and len(df) else ""
    ax.set_xlabel(f"Mean difference vs. reference ({metric}), 95% CI")
    ax.set_title("Statistical significance of pairwise comparisons\n(filled = significant after Holm-Bonferroni correction)")
    ax.grid(alpha=0.3, axis="x")
    return _finish(fig, save_path)


# ---------------------------------------------------------------------------
# Walk-forward (rolling-origin) temporal cross-validation
# ---------------------------------------------------------------------------

def plot_walk_forward_folds(fold_df: pd.DataFrame, summary_df: pd.DataFrame | None = None,
                             metric: str = "qpu_yield_pct", save_path: str | None = None) -> Figure:
    """
    Bar chart of `metric` per walk-forward fold (from
    `walk_forward.run_walk_forward_evaluation`'s `fold_df`), with a
    horizontal line at the aggregate mean and a shaded band at its 95%
    CI (from `summary_df`, if provided) -- shows at a glance whether the
    metric is stable across folds (tight scatter around the mean) or
    drifts/varies substantially across different slices of time.
    """
    fig, ax = plt.subplots(figsize=(max(6, 1.1 * len(fold_df)), 4.5))
    x = fold_df["fold"].to_numpy()
    values = fold_df[metric].to_numpy()
    ax.bar(x, values, color="tab:blue", alpha=0.8)

    if summary_df is not None and "Metric" in summary_df.columns:
        row = summary_df[summary_df["Metric"] == metric]
        if not row.empty:
            mean = row["Mean"].iloc[0]
            lo = row["95% CI Low"].iloc[0]
            hi = row["95% CI High"].iloc[0]
            ax.axhline(mean, color="tab:red", linestyle="--", linewidth=1.5, label=f"Mean = {mean:.3g}")
            if lo == lo and hi == hi:  # not NaN
                ax.axhspan(lo, hi, color="tab:red", alpha=0.15, label="95% CI")
            ax.legend(loc="best", fontsize=8)

    ax.set_xticks(x)
    ax.set_xlabel("Walk-forward fold")
    ax.set_ylabel(metric)
    ax.set_title(f"{metric} across walk-forward folds")
    ax.grid(alpha=0.3, axis="y")
    return _finish(fig, save_path)


def plot_walk_forward_timeline(splits: list[tuple[range, range]], n_samples: int | None = None,
                                save_path: str | None = None) -> Figure:
    """
    Horizontal timeline diagram of every walk-forward fold's train/test
    segments (from `walk_forward.generate_walk_forward_splits`'s output),
    one row per fold -- makes the "folds move strictly forward, never
    overlapping in their test segments" guarantee visually obvious.
    """
    fig, ax = plt.subplots(figsize=(9, max(2, 0.6 * len(splits))))
    for i, (train_range, test_range) in enumerate(splits):
        ax.barh(i, len(train_range), left=train_range.start, color="tab:blue", alpha=0.7, label="Train" if i == 0 else None)
        ax.barh(i, len(test_range), left=test_range.start, color="tab:orange", alpha=0.9, label="Test" if i == 0 else None)

    ax.set_yticks(range(len(splits)))
    ax.set_yticklabels([f"Fold {i}" for i in range(len(splits))])
    ax.invert_yaxis()
    ax.set_xlabel("Sample index (chronological order)")
    ax.set_title("Walk-forward fold timeline")
    if n_samples is not None:
        ax.set_xlim(0, n_samples)
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(alpha=0.3, axis="x")
    return _finish(fig, save_path)
