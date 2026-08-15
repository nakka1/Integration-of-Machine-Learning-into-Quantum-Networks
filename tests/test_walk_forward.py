import pytest
import torch

from quantum_twin.config import TrainConfig, QuantumConfig, WalkForwardConfig
from quantum_twin.walk_forward import generate_walk_forward_splits, run_walk_forward_evaluation


# ---------------------------------------------------------------------------
# generate_walk_forward_splits: pure logic, no training involved
# ---------------------------------------------------------------------------

def test_expanding_window_folds_are_non_overlapping_and_forward_moving():
    wf_cfg = WalkForwardConfig(n_splits=5, test_size=150, min_train_size=300, expanding=True)
    n_samples = 300 + 5 * 150  # exactly fits 5 folds
    splits = generate_walk_forward_splits(n_samples, wf_cfg)

    assert len(splits) == 5
    for train_range, test_range in splits:
        assert train_range.start == 0  # expanding always starts at sample 0
        assert len(test_range) == 150

    for k in range(1, 5):
        prev_test, cur_test = splits[k - 1][1], splits[k][1]
        assert cur_test.start >= prev_test.stop  # non-overlapping, forward-moving
        assert splits[k][0].stop > splits[k - 1][0].stop  # training set grows


def test_rolling_window_folds_have_fixed_training_size():
    wf_cfg = WalkForwardConfig(n_splits=5, test_size=150, min_train_size=300, expanding=False)
    n_samples = 300 + 5 * 150
    splits = generate_walk_forward_splits(n_samples, wf_cfg)
    for train_range, _test_range in splits:
        assert len(train_range) == 300


def test_gap_is_respected_between_train_and_test_segments():
    wf_cfg = WalkForwardConfig(n_splits=2, test_size=100, min_train_size=200, gap=20)
    splits = generate_walk_forward_splits(1000, wf_cfg)
    for train_range, test_range in splits:
        assert test_range.start - train_range.stop == 20


def test_insufficient_data_raises_value_error():
    wf_cfg = WalkForwardConfig(n_splits=5, test_size=150, min_train_size=300)
    with pytest.raises(ValueError):
        generate_walk_forward_splits(100, wf_cfg)


def test_partial_data_returns_fewer_folds_than_requested():
    wf_cfg = WalkForwardConfig(n_splits=5, test_size=150, min_train_size=300)
    n_samples = 300 + 2 * 150 + 50  # fits exactly 2 folds, not a 3rd
    splits = generate_walk_forward_splits(n_samples, wf_cfg)
    assert len(splits) == 2


def test_single_fold_configuration():
    wf_cfg = WalkForwardConfig(n_splits=1, test_size=50, min_train_size=100)
    splits = generate_walk_forward_splits(150, wf_cfg)
    assert len(splits) == 1
    train_range, test_range = splits[0]
    assert train_range == range(0, 100)
    assert test_range == range(100, 150)


# ---------------------------------------------------------------------------
# run_walk_forward_evaluation: light structural / smoke test (tiny epochs)
# ---------------------------------------------------------------------------

def test_run_walk_forward_evaluation_returns_expected_shapes():
    torch.manual_seed(0)
    device = torch.device("cpu")

    n_samples = 80
    X_full = torch.rand(n_samples, 10, 2)
    y_full = torch.rand(n_samples, 1)

    train_cfg = TrainConfig(hidden_size=4, epochs=3, threshold=0.65)
    quantum_cfg = QuantumConfig(shots=8)
    wf_cfg = WalkForwardConfig(n_splits=2, test_size=15, min_train_size=30, epochs=3, lr=0.01, seed=0)

    fold_df, summary_df, splits = run_walk_forward_evaluation(
        X_full, y_full, device, train_cfg=train_cfg, quantum_cfg=quantum_cfg, wf_cfg=wf_cfg,
    )

    assert len(fold_df) == len(splits)
    assert len(fold_df) >= 1
    assert {"fold", "mae", "rmse", "r2", "qpu_yield_pct", "fp", "fn"}.issubset(fold_df.columns)
    assert not summary_df.empty
    assert {"Metric", "Mean", "95% CI Low", "95% CI High", "N Folds"}.issubset(summary_df.columns)


def test_run_walk_forward_evaluation_folds_use_distinct_seeds():
    # Each fold offsets WalkForwardConfig.seed by its own index -- verify
    # this doesn't crash and produces folds (a weak but real smoke check
    # that the seed-offset wiring is at least syntactically exercised).
    torch.manual_seed(0)
    device = torch.device("cpu")
    n_samples = 70
    X_full = torch.rand(n_samples, 8, 2)
    y_full = torch.rand(n_samples, 1)

    train_cfg = TrainConfig(hidden_size=4, epochs=2)
    quantum_cfg = QuantumConfig(shots=8)
    wf_cfg = WalkForwardConfig(n_splits=3, test_size=10, min_train_size=20, epochs=2, seed=100)

    fold_df, _summary_df, splits = run_walk_forward_evaluation(
        X_full, y_full, device, train_cfg=train_cfg, quantum_cfg=quantum_cfg, wf_cfg=wf_cfg,
    )
    assert len(splits) == 3
    assert list(fold_df["fold"]) == list(range(len(splits)))
