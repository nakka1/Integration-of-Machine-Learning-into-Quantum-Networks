import pytest
import torch

from qrepeater_twin.ablation import _build_decomposition_df, run_ablation_study
from qrepeater_twin.config import AblationConfig, QuantumConfig, TrainConfig


# ---------------------------------------------------------------------------
# _build_decomposition_df: pure arithmetic, no training involved
# ---------------------------------------------------------------------------

def test_build_decomposition_df_matches_manual_factorial_formulas():
    cell_metric_means = {
        ("EdgeLSTM", "MSE"): {"qpu_yield_pct": 60.0, "mae": 0.05},
        ("EdgeLSTM", "CS-MSE"): {"qpu_yield_pct": 90.0, "mae": 0.04},
        ("StandardLSTM", "MSE"): {"qpu_yield_pct": 55.0, "mae": 0.06},
        ("StandardLSTM", "CS-MSE"): {"qpu_yield_pct": 65.0, "mae": 0.055},
    }
    df = _build_decomposition_df(cell_metric_means, ["qpu_yield_pct", "mae"])

    row = df[df["Metric"] == "qpu_yield_pct"].iloc[0]
    # Architecture effect = ((60+90)/2) - ((55+65)/2) = 75 - 60 = 15
    assert row["Architecture Effect"] == pytest.approx(15.0)
    # Loss effect = ((90+65)/2) - ((60+55)/2) = 77.5 - 57.5 = 20
    assert row["Loss Effect"] == pytest.approx(20.0)
    # Interaction = (90-60) - (65-55) = 30 - 10 = 20
    assert row["Interaction Effect"] == pytest.approx(20.0)


def test_build_decomposition_df_zero_interaction_when_purely_additive():
    # Architecture adds +10 regardless of loss; loss adds +5 regardless of
    # architecture -> perfectly additive, interaction must be exactly 0.
    cell_metric_means = {
        ("EdgeLSTM", "MSE"): {"metric": 100.0},
        ("EdgeLSTM", "CS-MSE"): {"metric": 105.0},
        ("StandardLSTM", "MSE"): {"metric": 90.0},
        ("StandardLSTM", "CS-MSE"): {"metric": 95.0},
    }
    df = _build_decomposition_df(cell_metric_means, ["metric"])
    row = df.iloc[0]
    assert row["Interaction Effect"] == pytest.approx(0.0, abs=1e-9)
    assert row["Architecture Effect"] == pytest.approx(10.0)
    assert row["Loss Effect"] == pytest.approx(5.0)


def test_build_decomposition_df_interpretation_mentions_additivity_when_small():
    cell_metric_means = {
        ("EdgeLSTM", "MSE"): {"metric": 100.0},
        ("EdgeLSTM", "CS-MSE"): {"metric": 105.0},
        ("StandardLSTM", "MSE"): {"metric": 90.0},
        ("StandardLSTM", "CS-MSE"): {"metric": 95.0},
    }
    df = _build_decomposition_df(cell_metric_means, ["metric"])
    assert "additive" in df.iloc[0]["Interpretation"].lower()


def test_build_decomposition_df_interpretation_flags_lower_is_better_correctly():
    # For 'mae' (lower is better): EdgeLSTM has LOWER mae than StandardLSTM
    # at both loss settings -> architecture effect is negative (Edge - Std < 0)
    # and must be reported as an IMPROVEMENT, not a regression.
    cell_metric_means = {
        ("EdgeLSTM", "MSE"): {"mae": 0.03},
        ("EdgeLSTM", "CS-MSE"): {"mae": 0.02},
        ("StandardLSTM", "MSE"): {"mae": 0.05},
        ("StandardLSTM", "CS-MSE"): {"mae": 0.045},
    }
    df = _build_decomposition_df(cell_metric_means, ["mae"])
    row = df.iloc[0]
    assert row["Architecture Effect"] < 0  # Edge's mae is lower (better)
    assert "improves" in row["Interpretation"].lower()


# ---------------------------------------------------------------------------
# run_ablation_study: light structural / smoke test (tiny epochs, 1 seed)
# ---------------------------------------------------------------------------

def test_run_ablation_study_returns_all_four_grid_cells():
    torch.manual_seed(0)
    device = torch.device("cpu")

    X_train = torch.rand(24, 10, 2)
    y_train = torch.rand(24, 1)
    X_test = torch.rand(12, 10, 2)
    y_test = torch.rand(12, 1)

    train_cfg = TrainConfig(hidden_size=4, epochs=3, threshold=0.65)
    quantum_cfg = QuantumConfig(shots=8)
    ablation_cfg = AblationConfig(
        standard_lstm_hidden_size=6, standard_lstm_num_layers=1, standard_lstm_dropout=0.0,
        epochs=3, lr=0.01, representative_lambda=5.0, seeds=[0],
    )

    results_df, decomposition_df, baseline_metrics, per_cell_seed_results = run_ablation_study(
        X_train, y_train, X_test, y_test, device,
        train_cfg=train_cfg, quantum_cfg=quantum_cfg, ablation_cfg=ablation_cfg,
    )

    assert len(results_df) == 4
    assert set(zip(results_df["Architecture"], results_df["Loss"])) == {
        ("EdgeLSTM", "MSE"), ("EdgeLSTM", "CS-MSE"),
        ("StandardLSTM", "MSE"), ("StandardLSTM", "CS-MSE"),
    }
    assert len(decomposition_df) == len(ablation_cfg.headline_metrics)
    assert set(per_cell_seed_results.keys()) == {
        ("EdgeLSTM", "MSE"), ("EdgeLSTM", "CS-MSE"),
        ("StandardLSTM", "MSE"), ("StandardLSTM", "CS-MSE"),
    }
    assert baseline_metrics["total_steps"] == 12
