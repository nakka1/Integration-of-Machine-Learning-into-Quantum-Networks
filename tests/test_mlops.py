"""
Tests for `quantum_twin.mlops`, covering both operating modes:

    1. `mlflow` genuinely absent (the common case in this sandbox, and
       for anyone who hasn't installed the optional `[mlflow]` extra) --
       every `MLflowTracker` method must be a safe no-op.
    2. `mlflow` present (simulated here via a `MagicMock` injected into
       `sys.modules["mlflow"]` BEFORE `quantum_twin.mlops` is imported,
       since real `mlflow` is not installed in this test environment
       either) -- every method must call the correct underlying `mlflow.*`
       function with the correct arguments.

The `fake_mlflow_module` fixture performs the `sys.modules` injection +
fresh re-import needed for case 2; case 1 uses the plain top-level
`quantum_twin.mlops` import (real absence, no patching needed) since that
already reflects this environment's actual state.
"""

from __future__ import annotations

import dataclasses
import sys
import warnings
from unittest.mock import MagicMock

import pandas as pd
import pytest


# ---------------------------------------------------------------------------
# Case 1: mlflow genuinely absent -- safe no-op path
# ---------------------------------------------------------------------------

def test_mlflow_available_flag_reflects_real_environment():
    from quantum_twin import mlops

    # In this test environment mlflow is not installed; MLFLOW_AVAILABLE
    # must honestly reflect that rather than hard-coding True.
    import importlib.util

    real_mlflow_installed = importlib.util.find_spec("mlflow") is not None
    assert mlops.MLFLOW_AVAILABLE == real_mlflow_installed


def test_tracker_methods_are_safe_noops_without_mlflow():
    from quantum_twin import mlops

    if mlops.MLFLOW_AVAILABLE:
        pytest.skip("mlflow is actually installed in this environment; no-op path not exercised.")

    tracker = mlops.MLflowTracker("test_exp", run_name="run1")
    # None of the following may raise, regardless of mlflow's absence.
    tracker.start()
    tracker.log_config({"a": 1})
    tracker.log_metrics({"mae": 0.05})
    tracker.log_table(pd.DataFrame({"x": [1, 2]}), "table1")
    fig = MagicMock()
    tracker.log_figure(fig, "chart")
    tracker.log_latex_table(pd.DataFrame({"x": [1]}), "table2")
    tracker.end()


def test_warns_exactly_once_per_process_not_per_call():
    from quantum_twin import mlops

    if mlops.MLFLOW_AVAILABLE:
        pytest.skip("mlflow is actually installed in this environment; no-op path not exercised.")

    mlops._WARNED_ONCE = False  # reset so this test is order-independent
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        tracker = mlops.MLflowTracker("exp")
        tracker.start()
        tracker.log_config({"a": 1})
        tracker.log_metrics({"mae": 0.1})
        relevant = [w for w in caught if "mlflow" in str(w.message).lower()]
        assert len(relevant) == 1


# ---------------------------------------------------------------------------
# Pure logic: _flatten_config (no mlflow involvement at all)
# ---------------------------------------------------------------------------

def test_flatten_config_nested_dicts():
    from quantum_twin.mlops import _flatten_config

    flat = _flatten_config({"sim_config": {"n_steps": 4000, "seed": 42}, "device": "cpu"})
    assert flat == {"sim_config.n_steps": 4000, "sim_config.seed": 42, "device": "cpu"}


def test_flatten_config_stringifies_lists():
    from quantum_twin.mlops import _flatten_config

    flat = _flatten_config({"seeds": [42, 43, 44]})
    assert flat == {"seeds": "[42, 43, 44]"}


def test_flatten_config_deeply_nested():
    from quantum_twin.mlops import _flatten_config

    flat = _flatten_config({"a": {"b": {"c": 1}}})
    assert flat == {"a.b.c": 1}


# ---------------------------------------------------------------------------
# Case 2: mlflow present (faked via sys.modules injection + re-import)
# ---------------------------------------------------------------------------

@pytest.fixture
def fake_mlflow_module(monkeypatch):
    """
    Injects a `MagicMock` as `sys.modules["mlflow"]`, then forces a FRESH
    import of `quantum_twin.mlops` (removing any cached import first) so
    its `try: import mlflow` picks up the fake -- yields
    `(mlops_module, fake_mlflow_mock)`.

    Cleans up by removing both fake modules from `sys.modules` afterward,
    so later tests (including
    `test_mlflow_available_flag_reflects_real_environment` above) see the
    REAL absence of mlflow again, undisturbed by this fixture having run.
    """
    fake_mlflow = MagicMock()
    monkeypatch.setitem(sys.modules, "mlflow", fake_mlflow)
    monkeypatch.delitem(sys.modules, "quantum_twin.mlops", raising=False)

    import importlib

    mlops_module = importlib.import_module("quantum_twin.mlops")
    yield mlops_module, fake_mlflow

    # Explicitly remove the fake "mlflow" entry HERE, before re-importing
    # quantum_twin.mlops -- relying on monkeypatch's own automatic
    # teardown to remove it LATER would leave sys.modules["mlflow"]
    # pointing at this fixture's fake mlflow for the re-import below
    # (pytest tears fixtures down in reverse dependency order, so
    # `monkeypatch`'s automatic undo runs AFTER this generator's
    # post-yield code, not before it) -- re-importing while the fake is
    # still registered would silently produce another mlflow-PRESENT
    # module instead of restoring the real absence this fixture promises
    # to leave behind for later tests.
    monkeypatch.delitem(sys.modules, "mlflow", raising=False)
    monkeypatch.delitem(sys.modules, "quantum_twin.mlops", raising=False)
    importlib.import_module("quantum_twin.mlops")  # restore the real-absence version for later tests


def test_start_calls_set_experiment_and_start_run(fake_mlflow_module):
    mlops_module, fake_mlflow = fake_mlflow_module
    assert mlops_module.MLFLOW_AVAILABLE is True

    tracker = mlops_module.MLflowTracker("pareto_sweep", run_name="run1")
    tracker.start()

    fake_mlflow.set_experiment.assert_called_once_with("pareto_sweep")
    fake_mlflow.start_run.assert_called_once_with(run_name="run1")


def test_log_config_flattens_dataclasses(fake_mlflow_module):
    mlops_module, fake_mlflow = fake_mlflow_module

    @dataclasses.dataclass
    class DummyCfg:
        a: int = 1
        b: str = "x"

    tracker = mlops_module.MLflowTracker("exp")
    tracker.log_config({"sim_config": DummyCfg(), "device": "cpu"})

    logged = fake_mlflow.log_params.call_args[0][0]
    assert logged == {"sim_config.a": 1, "sim_config.b": "x", "device": "cpu"}


def test_log_metrics_drops_non_numeric_values(fake_mlflow_module):
    mlops_module, fake_mlflow = fake_mlflow_module

    tracker = mlops_module.MLflowTracker("exp")
    tracker.log_metrics({"mae": 0.05, "note": "not_numeric"})

    logged = fake_mlflow.log_metrics.call_args[0][0]
    assert logged == {"mae": 0.05}


def test_log_table_logs_a_csv_artifact(fake_mlflow_module):
    mlops_module, fake_mlflow = fake_mlflow_module

    tracker = mlops_module.MLflowTracker("exp")
    tracker.log_table(pd.DataFrame({"x": [1, 2, 3]}), "results")

    fake_mlflow.log_artifact.assert_called()
    logged_path = fake_mlflow.log_artifact.call_args[0][0]
    assert logged_path.endswith("results.csv")


def test_log_figure_calls_mlflow_log_figure(fake_mlflow_module):
    mlops_module, fake_mlflow = fake_mlflow_module

    fig = MagicMock()
    tracker = mlops_module.MLflowTracker("exp")
    tracker.log_figure(fig, "chart")

    fake_mlflow.log_figure.assert_called_once_with(fig, "chart.png")


def test_end_calls_mlflow_end_run(fake_mlflow_module):
    mlops_module, fake_mlflow = fake_mlflow_module

    tracker = mlops_module.MLflowTracker("exp")
    tracker.end()

    fake_mlflow.end_run.assert_called_once()


def test_context_manager_starts_and_ends_run(fake_mlflow_module):
    mlops_module, fake_mlflow = fake_mlflow_module

    with mlops_module.MLflowTracker("exp"):
        pass

    fake_mlflow.start_run.assert_called_once()
    fake_mlflow.end_run.assert_called_once()
