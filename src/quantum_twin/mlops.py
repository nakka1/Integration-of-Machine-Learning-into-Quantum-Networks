r"""
Component 16 -- Professional experiment tracking via MLflow (optional).

`experiment_tracking.ExperimentRun` (plain CSV/PNG/JSON/TeX files on
disk, no external service) remains this project's DEFAULT, always-on
tracking backend -- it has zero setup cost and works offline, which
matters for a research pipeline that needs to run anywhere. This module
adds an OPT-IN second backend, MLflow, for teams that already run an
MLflow tracking server (or want side-by-side run comparison, a
searchable run history, and model-registry integration that flat files
on disk don't give you for free).

`mlflow` is declared as an OPTIONAL dependency (`pip install
"quantum-twin[mlflow]"`, see `pyproject.toml`), imported behind a
try/except at module load time. Every public function here checks
`MLFLOW_AVAILABLE` first and degrades to a no-op (with a single printed
notice, not a raised exception) when it is `False` -- calling this
module's functions in an environment without `mlflow` installed is
always safe; it simply logs nothing to MLflow while the rest of the
pipeline (including `experiment_tracking.ExperimentRun`) continues
exactly as before. This is a deliberate design choice: an optional
observability integration must never be able to crash the actual
experiment it is trying to observe.
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any

import pandas as pd

try:
    import mlflow  # type: ignore[import-untyped]

    MLFLOW_AVAILABLE = True
except ImportError:  # pragma: no cover - exercised via mocked import in tests
    mlflow = None  # type: ignore[assignment]
    MLFLOW_AVAILABLE = False

_WARNED_ONCE = False


def _warn_mlflow_unavailable() -> None:
    """Prints the "MLflow isn't installed" notice exactly once per
    process (not once per logging call), so a long experiment loop that
    calls `log_metrics` hundreds of times doesn't spam the console."""
    global _WARNED_ONCE
    if not _WARNED_ONCE:
        warnings.warn(
            "quantum_twin.mlops: mlflow is not installed -- MLflow tracking calls are "
            "no-ops for this run. Install with `pip install \"quantum-twin[mlflow]\"` to "
            "enable it; the local ExperimentRun backend (experiment_tracking.py) is "
            "unaffected and remains the tracking source of truth either way.",
            stacklevel=2,
        )
        _WARNED_ONCE = True


def _flatten_config(config: dict[str, Any], parent_key: str = "") -> dict[str, Any]:
    """
    MLflow params/metrics must be a FLAT mapping of primitive values;
    this project's configs are nested dataclasses-turned-dicts (e.g.
    `{"sim_config": {"n_steps": 4000, ...}, "train_config": {...}}`).
    Recursively flattens nested dicts into dotted keys
    (`"sim_config.n_steps"`), and stringifies lists (MLflow params are
    string-valued) -- mirrors the flattening `pandas.json_normalize`
    would do, without adding a dependency on it for this one operation.
    """
    flat: dict[str, Any] = {}
    for key, value in config.items():
        full_key = f"{parent_key}.{key}" if parent_key else key
        if isinstance(value, dict):
            flat.update(_flatten_config(value, full_key))
        elif isinstance(value, (list, tuple)):
            flat[full_key] = str(value)
        else:
            flat[full_key] = value
    return flat


class MLflowTracker:
    """
    Thin wrapper around one MLflow run's lifecycle
    (`mlflow.start_run()` / `mlflow.end_run()`), exposing the same small
    surface as `experiment_tracking.ExperimentRun`
    (`log_config`/`log_table`/`log_metrics`/`log_figure`/`log_latex_table`)
    so the two backends can be swapped or used side by side with minimal
    call-site duplication.

    Every method is a safe no-op (returns `None`, prints nothing beyond
    the one-time `MLFLOW_AVAILABLE` notice) when `mlflow` is not
    installed -- see the module docstring.

    Usage
    -----
        with MLflowTracker("pareto_sweep", run_name="lambda_sweep_v1") as tracker:
            tracker.log_config({"sim_config": sim_cfg, "train_config": train_cfg})
            tracker.log_table(results_df, "pareto_frontier")
            tracker.log_metrics({"best_qpu_yield_pct": 92.3})
            tracker.log_figure(fig, "pareto_frontier")
    """

    def __init__(self, experiment_name: str, run_name: str | None = None) -> None:
        self.experiment_name = experiment_name
        self.run_name = run_name
        self._active_run: Any = None

    def __enter__(self) -> "MLflowTracker":
        self.start()
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.end()

    def start(self) -> None:
        if not MLFLOW_AVAILABLE:
            _warn_mlflow_unavailable()
            return
        mlflow.set_experiment(self.experiment_name)
        self._active_run = mlflow.start_run(run_name=self.run_name)

    def end(self) -> None:
        if not MLFLOW_AVAILABLE:
            return
        mlflow.end_run()
        self._active_run = None

    def log_config(self, config: dict[str, Any]) -> None:
        """Logs a (possibly nested) config dict as flat MLflow params.
        Dataclass instances are converted via `dataclasses.asdict` first,
        same convention as `experiment_tracking.ExperimentRun.save_config`."""
        if not MLFLOW_AVAILABLE:
            _warn_mlflow_unavailable()
            return
        import dataclasses

        serializable = {
            k: (dataclasses.asdict(v) if dataclasses.is_dataclass(v) else v) for k, v in config.items()
        }
        mlflow.log_params(_flatten_config(serializable))

    def log_metrics(self, metrics: dict[str, float], step: int | None = None) -> None:
        """Logs scalar metrics. Non-numeric values are silently dropped
        (MLflow's `log_metrics` requires floats) -- use `log_config` or a
        local `ExperimentRun.save_metrics` call for non-scalar metadata."""
        if not MLFLOW_AVAILABLE:
            _warn_mlflow_unavailable()
            return
        numeric_metrics = {k: float(v) for k, v in metrics.items() if isinstance(v, (int, float))}
        mlflow.log_metrics(numeric_metrics, step=step)

    def log_table(self, df: pd.DataFrame, name: str) -> None:
        """Logs a DataFrame as an MLflow artifact (CSV)."""
        if not MLFLOW_AVAILABLE:
            _warn_mlflow_unavailable()
            return
        mlflow.log_table(data=df, artifact_file=f"{name}.json") if hasattr(mlflow, "log_table") else None
        # `log_table` requires a fairly recent MLflow; a plain CSV artifact
        # is logged unconditionally too, so older MLflow servers still get
        # the data in a universally readable form.
        import tempfile

        with tempfile.TemporaryDirectory() as tmp_dir:
            csv_path = Path(tmp_dir) / f"{name}.csv"
            df.to_csv(csv_path, index=False)
            mlflow.log_artifact(str(csv_path))

    def log_figure(self, fig: Any, name: str) -> None:
        """Logs a `matplotlib.figure.Figure` as an MLflow artifact (PNG)."""
        if not MLFLOW_AVAILABLE:
            _warn_mlflow_unavailable()
            return
        mlflow.log_figure(fig, f"{name}.png")

    def log_latex_table(self, df: pd.DataFrame, name: str, caption: str | None = None,
                         label: str | None = None) -> None:
        """Logs a `.tex` rendering of `df` (via `latex_export.dataframe_to_latex`)
        as an MLflow artifact."""
        if not MLFLOW_AVAILABLE:
            _warn_mlflow_unavailable()
            return
        from .latex_export import dataframe_to_latex

        import tempfile

        latex_source = dataframe_to_latex(df, caption=caption, label=label)
        with tempfile.TemporaryDirectory() as tmp_dir:
            tex_path = Path(tmp_dir) / f"{name}.tex"
            tex_path.write_text(latex_source)
            mlflow.log_artifact(str(tex_path))
