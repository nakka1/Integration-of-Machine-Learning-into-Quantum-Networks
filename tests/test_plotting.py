import matplotlib
matplotlib.use("Agg")  # headless backend for test environments without a display

import numpy as np
import pandas as pd
import pytest
from matplotlib.figure import Figure

from quantum_twin import plotting


# ---------------------------------------------------------------------------
# _parse_mean_std_column
# ---------------------------------------------------------------------------

def test_parse_mean_std_column_basic():
    series = pd.Series(["12.34 +/- 1.02", "-5.0 +/- 0.3", "0.0000 +/- 0.0001"])
    means, stds = plotting._parse_mean_std_column(series)
    assert np.allclose(means, [12.34, -5.0, 0.0])
    assert np.allclose(stds, [1.02, 0.3, 0.0001])


def test_parse_mean_std_column_malformed_raises():
    series = pd.Series(["not a number"])
    with pytest.raises(ValueError):
        plotting._parse_mean_std_column(series)


# ---------------------------------------------------------------------------
# Every plotting function: smoke test that it returns a Figure and doesn't crash
# ---------------------------------------------------------------------------

def test_plot_pareto_frontier_returns_figure():
    df = pd.DataFrame({
        "Lambda": [1.0, 2.0, 5.0],
        "QPU Yield (%)": ["50.0 +/- 2.0", "60.0 +/- 1.5", "70.0 +/- 1.0"],
        "MAE": ["0.05 +/- 0.01", "0.04 +/- 0.008", "0.03 +/- 0.005"],
    })
    fig = plotting.plot_pareto_frontier(df)
    assert isinstance(fig, Figure)


def test_plot_model_comparison_bars_returns_figure():
    df = pd.DataFrame({"Model": ["A", "B", "C"], "QPU Yield (%)": ["50 +/- 2", "60 +/- 1", "55 +/- 3"]})
    fig = plotting.plot_model_comparison_bars(df)
    assert isinstance(fig, Figure)


def test_plot_confusion_matrix_returns_figure():
    fig = plotting.plot_confusion_matrix({"TP": 80, "FP": 20, "TN": 150, "FN": 30})
    assert isinstance(fig, Figure)


def test_plot_decision_matrix_returns_figure():
    dm = pd.DataFrame({"Model": ["A", "B"], "Decision Score": [0.8, 0.5], "Rank": [1, 2]})
    fig = plotting.plot_decision_matrix(dm)
    assert isinstance(fig, Figure)


def test_plot_sensitivity_summary_returns_figure():
    sm = pd.DataFrame({"Model": ["A", "B"], "Win Rate (%)": [100.0, 20.0]})
    fig = plotting.plot_sensitivity_summary(sm)
    assert isinstance(fig, Figure)


def test_plot_temporal_prediction_error_returns_figure_with_events():
    tm = {"timing_errors_steps": [1.0, -2.0, 0.5, 3.0], "n_missed_events": 1, "n_false_alarms": 0}
    fig = plotting.plot_temporal_prediction_error(tm)
    assert isinstance(fig, Figure)


def test_plot_temporal_prediction_error_handles_no_matched_events():
    # No matched events at all (e.g. predictor missed every degradation) --
    # must not crash on an empty histogram.
    tm = {"timing_errors_steps": [], "n_missed_events": 3, "n_false_alarms": 0}
    fig = plotting.plot_temporal_prediction_error(tm)
    assert isinstance(fig, Figure)


def test_plot_fidelity_timeseries_with_crossings_returns_figure():
    y_true = np.linspace(1, 0, 50)
    y_pred = y_true + 0.02
    fig = plotting.plot_fidelity_timeseries_with_crossings(
        y_true, y_pred, 0.65, true_crossings=[17.5], pred_crossings=[16.8],
    )
    assert isinstance(fig, Figure)


def test_plot_fidelity_timeseries_with_crossings_handles_no_crossings():
    y_true = np.ones(20) * 0.9
    y_pred = np.ones(20) * 0.9
    fig = plotting.plot_fidelity_timeseries_with_crossings(y_true, y_pred, 0.65)
    assert isinstance(fig, Figure)


def test_plot_ablation_interaction_returns_figure():
    decomp = pd.DataFrame({
        "Metric": ["qpu_yield_pct"],
        "EdgeLSTM+MSE": [60.0], "EdgeLSTM+CS-MSE": [90.0],
        "StandardLSTM+MSE": [55.0], "StandardLSTM+CS-MSE": [65.0],
        "Architecture Effect": [15.0], "Loss Effect": [20.0], "Interaction Effect": [20.0],
        "Interpretation": ["test"],
    })
    fig = plotting.plot_ablation_interaction(decomp, "qpu_yield_pct")
    assert isinstance(fig, Figure)


def test_plot_ablation_interaction_unknown_metric_raises():
    decomp = pd.DataFrame({
        "Metric": ["qpu_yield_pct"],
        "EdgeLSTM+MSE": [60.0], "EdgeLSTM+CS-MSE": [90.0],
        "StandardLSTM+MSE": [55.0], "StandardLSTM+CS-MSE": [65.0],
        "Architecture Effect": [15.0], "Loss Effect": [20.0], "Interaction Effect": [20.0],
        "Interpretation": ["test"],
    })
    with pytest.raises(ValueError):
        plotting.plot_ablation_interaction(decomp, "nonexistent_metric")


def test_save_path_actually_writes_a_file(tmp_path):
    df = pd.DataFrame({"Model": ["A", "B"], "QPU Yield (%)": ["50 +/- 2", "60 +/- 1"]})
    save_path = tmp_path / "chart.png"
    plotting.plot_model_comparison_bars(df, save_path=str(save_path))
    assert save_path.exists()
    assert save_path.stat().st_size > 0


# ---------------------------------------------------------------------------
# New in v3.3: significance forest plot, walk-forward plots
# ---------------------------------------------------------------------------

def test_plot_significance_forest_returns_figure():
    sig_df = pd.DataFrame({
        "Model": ["LSTM+MSE", "RandomForest", "Oracle"],
        "Metric": ["qpu_yield_pct"] * 3,
        "Mean Difference": [-25.0, -15.0, 14.0],
        "95% CI Low": [-30.0, -20.0, 10.0],
        "95% CI High": [-20.0, -10.0, 18.0],
        "Significant (t-test, Holm, alpha=0.05)": [True, True, False],
    })
    fig = plotting.plot_significance_forest(sig_df)
    assert isinstance(fig, Figure)


def test_plot_significance_forest_handles_missing_significance_column():
    sig_df = pd.DataFrame({
        "Model": ["LSTM+MSE"],
        "Metric": ["mae"],
        "Mean Difference": [0.02],
        "95% CI Low": [0.01],
        "95% CI High": [0.03],
    })
    fig = plotting.plot_significance_forest(sig_df)
    assert isinstance(fig, Figure)


def test_plot_walk_forward_folds_returns_figure_with_summary():
    fold_df = pd.DataFrame({"fold": [0, 1, 2], "qpu_yield_pct": [80.0, 82.0, 78.0]})
    summary_df = pd.DataFrame({
        "Metric": ["qpu_yield_pct"], "Mean": [80.0],
        "95% CI Low": [76.0], "95% CI High": [84.0], "N Folds": [3],
    })
    fig = plotting.plot_walk_forward_folds(fold_df, summary_df, metric="qpu_yield_pct")
    assert isinstance(fig, Figure)


def test_plot_walk_forward_folds_without_summary():
    fold_df = pd.DataFrame({"fold": [0, 1], "mae": [0.05, 0.04]})
    fig = plotting.plot_walk_forward_folds(fold_df, metric="mae")
    assert isinstance(fig, Figure)


def test_plot_walk_forward_timeline_returns_figure():
    splits = [
        (range(0, 300), range(300, 450)),
        (range(0, 450), range(450, 600)),
        (range(0, 600), range(600, 750)),
    ]
    fig = plotting.plot_walk_forward_timeline(splits, n_samples=800)
    assert isinstance(fig, Figure)
