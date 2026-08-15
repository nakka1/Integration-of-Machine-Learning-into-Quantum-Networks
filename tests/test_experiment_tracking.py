import dataclasses
import json

import pandas as pd
import pytest

from quantum_twin.experiment_tracking import (
    ExperimentRun,
    track_ablation_experiment,
    track_model_comparison_experiment,
    track_pareto_sweep_experiment,
)


@dataclasses.dataclass
class _DummyConfig:
    a: int = 1
    b: str = "x"


# ---------------------------------------------------------------------------
# ExperimentRun: low-level save_* / finalize() behavior
# ---------------------------------------------------------------------------

def test_experiment_run_creates_timestamped_directory(tmp_path):
    exp = ExperimentRun("my_experiment", base_dir=str(tmp_path))
    assert exp.dir.exists()
    assert exp.dir.parent == tmp_path
    assert exp.dir.name.startswith("my_experiment_")


def test_experiment_run_two_instances_get_distinct_directories(tmp_path):
    exp1 = ExperimentRun("same_name", base_dir=str(tmp_path))
    exp2 = ExperimentRun("same_name", base_dir=str(tmp_path))
    # Directories differ (by timestamp) even for identical names -- no run
    # ever silently overwrites a previous one.
    assert exp1.dir != exp2.dir


def test_save_config_serializes_dataclasses_and_plain_values(tmp_path):
    exp = ExperimentRun("cfg_test", base_dir=str(tmp_path))
    exp.save_config({"sim_config": _DummyConfig(), "device": "cpu", "note": "hello"})

    with open(exp.dir / "config.json") as f:
        loaded = json.load(f)

    assert loaded == {"sim_config": {"a": 1, "b": "x"}, "device": "cpu", "note": "hello"}


def test_save_table_writes_readable_csv(tmp_path):
    exp = ExperimentRun("table_test", base_dir=str(tmp_path))
    df = pd.DataFrame({"Lambda": [1, 2], "Yield": ["50 +/- 1", "60 +/- 2"]})
    exp.save_table(df, "results.csv")

    reloaded = pd.read_csv(exp.dir / "results.csv")
    assert list(reloaded["Lambda"]) == [1, 2]
    assert list(reloaded["Yield"]) == ["50 +/- 1", "60 +/- 2"]


def test_save_metrics_writes_valid_json(tmp_path):
    exp = ExperimentRun("metrics_test", base_dir=str(tmp_path))
    exp.save_metrics({"total_steps": 100, "useful_pairs": 42})

    with open(exp.dir / "metrics.json") as f:
        loaded = json.load(f)
    assert loaded == {"total_steps": 100, "useful_pairs": 42}


def test_finalize_manifest_lists_every_saved_file(tmp_path):
    exp = ExperimentRun("manifest_test", base_dir=str(tmp_path))
    exp.save_config({"a": 1})
    exp.save_table(pd.DataFrame({"x": [1]}), "t.csv")
    exp.save_metrics({"m": 1})
    manifest_path = exp.finalize()

    with open(manifest_path) as f:
        manifest = json.load(f)

    filenames = {entry["filename"] for entry in manifest["files"]}
    assert filenames == {"config.json", "t.csv", "metrics.json"}
    assert manifest["name"] == "manifest_test"


def test_save_metrics_handles_numpy_scalars_without_crashing(tmp_path):
    np = pytest.importorskip("numpy")
    exp = ExperimentRun("numpy_test", base_dir=str(tmp_path))
    exp.save_metrics({"mae": np.float64(0.05), "count": np.int64(42)})
    with open(exp.dir / "metrics.json") as f:
        loaded = json.load(f)
    assert loaded["mae"] == pytest.approx(0.05)
    assert loaded["count"] == 42


# ---------------------------------------------------------------------------
# track_pareto_sweep_experiment / track_model_comparison_experiment / track_ablation_experiment
# ---------------------------------------------------------------------------

def test_track_pareto_sweep_experiment_writes_expected_artifacts(tmp_path):
    results_df = pd.DataFrame({
        "Lambda": [1.0, 2.0],
        "QPU Yield (%)": ["50.0 +/- 2.0", "60.0 +/- 1.0"],
        "MAE": ["0.05 +/- 0.01", "0.03 +/- 0.005"],
    })
    baseline_metrics = {"total_steps": 100, "useful_pairs": 40, "attempted": 100,
                         "avg_classical_latency_s": 0.0}

    exp = track_pareto_sweep_experiment(
        "pareto_test", results_df, baseline_metrics,
        _DummyConfig(), _DummyConfig(), _DummyConfig(), _DummyConfig(), "cpu",
        base_dir=str(tmp_path),
    )

    files = {p.name for p in exp.dir.iterdir()}
    assert {"config.json", "pareto_frontier.csv", "pareto_frontier.png",
            "baseline_metrics.json", "manifest.json"} <= files


def test_track_model_comparison_experiment_writes_expected_artifacts(tmp_path):
    results_df = pd.DataFrame({"Model": ["EdgeLSTM+CS-MSE", "LSTM+MSE"],
                                "QPU Yield (%)": ["80 +/- 2", "60 +/- 3"]})
    decision_matrix_df = pd.DataFrame({"Model": ["EdgeLSTM+CS-MSE", "LSTM+MSE"],
                                        "Decision Score": [0.9, 0.5], "Rank": [1, 2]})
    sensitivity_summary_df = pd.DataFrame({"Model": ["EdgeLSTM+CS-MSE", "LSTM+MSE"],
                                            "Win Rate (%)": [100.0, 0.0]})
    sensitivity_trials_df = pd.DataFrame({"Winner": ["EdgeLSTM+CS-MSE"] * 4})
    baseline_metrics = {"total_steps": 100, "useful_pairs": 40}

    exp = track_model_comparison_experiment(
        "comparison_test", results_df, baseline_metrics, decision_matrix_df,
        (sensitivity_summary_df, sensitivity_trials_df, "verdict text"),
        _DummyConfig(), _DummyConfig(), _DummyConfig(), _DummyConfig(), _DummyConfig(), "cpu",
        base_dir=str(tmp_path),
    )

    files = {p.name for p in exp.dir.iterdir()}
    assert {"model_comparison.csv", "decision_matrix.csv", "decision_matrix.png",
            "sensitivity_summary.png", "sensitivity_trials.csv"} <= files


def test_track_ablation_experiment_writes_one_interaction_plot_per_metric(tmp_path):
    results_df = pd.DataFrame({
        "Model": ["EdgeLSTM+MSE", "EdgeLSTM+CS-MSE", "StandardLSTM+MSE", "StandardLSTM+CS-MSE"],
        "QPU Yield (%)": ["60 +/- 2", "90 +/- 1", "55 +/- 3", "65 +/- 2"],
    })
    decomposition_df = pd.DataFrame({
        "Metric": ["qpu_yield_pct", "mae"],
        "EdgeLSTM+MSE": [60.0, 0.05], "EdgeLSTM+CS-MSE": [90.0, 0.04],
        "StandardLSTM+MSE": [55.0, 0.06], "StandardLSTM+CS-MSE": [65.0, 0.055],
        "Architecture Effect": [15.0, -0.01], "Loss Effect": [20.0, -0.008],
        "Interaction Effect": [20.0, 0.002], "Interpretation": ["a", "b"],
    })
    baseline_metrics = {"total_steps": 100, "useful_pairs": 40}

    exp = track_ablation_experiment(
        "ablation_test", results_df, decomposition_df, baseline_metrics,
        _DummyConfig(), _DummyConfig(), _DummyConfig(), _DummyConfig(), "cpu",
        base_dir=str(tmp_path),
    )

    files = {p.name for p in exp.dir.iterdir()}
    assert "interaction_qpu_yield_pct.png" in files
    assert "interaction_mae.png" in files
    assert "ablation_decomposition.csv" in files
