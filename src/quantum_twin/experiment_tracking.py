r"""
Component 11 -- Automatic experiment-tracking pipeline (`ExperimentRun`).

Every run of `pareto_sweep.run_pareto_sweep`, `model_comparison.run_model_comparison`,
or `ablation.run_ablation_study` produces, on its own, only in-memory
Python objects (`DataFrame`s, `dict`s) -- nothing is written to disk, and
nothing records WHICH configuration (model, hyperparameters, epochs,
seeds, simulator/channel parameters) produced a given table of numbers.
Re-deriving that afterward from notebook scroll-back is exactly the
failure mode an automated experiment pipeline exists to prevent.

`ExperimentRun` wraps ONE experiment execution end-to-end: it creates a
timestamped output directory, then exposes small, focused `save_*`
methods that any of the three experiment types (or a bespoke script) can
call as they produce results. `track_pareto_sweep_experiment` /
`track_model_comparison_experiment` / `track_ablation_experiment` are
thin, ready-to-use wrappers around the three main experiment entry
points, each calling the matching `save_*` methods with the config
objects and result tables already in scope -- so a full, reproducible
experiment record is a single function call away, not something bolted
on after the fact.

Directory layout produced by one experiment (`<base_dir>/<name>_<timestamp>/`):

    config.json     -- every dataclass config used (asdict), plus the
                        device string and any extra metadata passed in.
    metrics.json     -- scalar/dict metrics not already inside a table
                         (e.g. the blind baseline's metrics dict).
    *.csv             -- every results/decomposition/decision-matrix
                          table, one file per table.
    *.tex              -- LaTeX renderings of the headline result tables
                           (via `latex_export.dataframe_to_latex`),
                           ready to `\input{}`/`\include{}` directly into
                           a written report.
    *.png               -- every figure generated for this experiment.
    manifest.json        -- lists every file written above, plus the
                             experiment name/timestamp, so the whole
                             directory can be audited or re-loaded without
                             re-running anything.
"""

from __future__ import annotations

import dataclasses
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    # Only needed for the `save_figure` type hint below -- matplotlib
    # stays an optional, function-local import everywhere else in this
    # module (see `track_*_experiment`'s `from . import plotting`) so
    # constructing an `ExperimentRun` never requires matplotlib to be
    # installed at all.
    from matplotlib.figure import Figure


def _json_default(obj):
    """`json.dumps(..., default=_json_default)`: makes numpy scalars,
    numpy arrays, and torch devices JSON-serializable without requiring
    every caller to pre-convert its metrics dict by hand."""
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return str(obj)  # last resort: torch.device, etc.


class ExperimentRun:
    """
    One tracked experiment execution: a timestamped directory plus
    small, focused methods to save configuration, tables, figures, and
    scalar metrics into it, and a `finalize()` that writes a manifest
    listing everything produced.

    Directory name: `<base_dir>/<name>_<YYYYmmdd_HHMMSS_ffffff>/` -- the
    microsecond-resolution timestamp guarantees repeated runs of the same
    experiment `name` never overwrite each other (even when created
    back-to-back within the same second, e.g. in a notebook loop), so
    every historical run stays on disk and reproducible.
    """

    def __init__(self, name: str, base_dir: str = "experiments") -> None:
        self.name = name
        self.timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
        self.dir = Path(base_dir) / f"{name}_{self.timestamp}"
        self.dir.mkdir(parents=True, exist_ok=True)
        self._manifest: Dict[str, Any] = {
            "name": name,
            "timestamp": self.timestamp,
            "directory": str(self.dir),
            "files": [],
        }

    def _register(self, filename: str, kind: str) -> Path:
        path = self.dir / filename
        self._manifest["files"].append({"filename": filename, "kind": kind})
        return path

    def save_config(self, config: Dict[str, Any], filename: str = "config.json") -> Path:
        """
        Saves a configuration dict to JSON. Dataclass instances anywhere
        in `config` (e.g. `SimConfig(...)`, `TrainConfig(...)`) are
        converted via `dataclasses.asdict` automatically, so callers can
        pass config OBJECTS directly rather than pre-serializing them:

            exp.save_config({"sim_config": sim_cfg, "train_config": train_cfg, "device": str(device)})
        """
        serializable = {
            k: (dataclasses.asdict(v) if dataclasses.is_dataclass(v) else v)
            for k, v in config.items()
        }
        path = self._register(filename, "config")
        with open(path, "w") as f:
            json.dump(serializable, f, indent=2, default=_json_default)
        return path

    def save_table(self, df: pd.DataFrame, filename: str) -> Path:
        """Saves a `pd.DataFrame` to CSV (`filename` should end in `.csv`)."""
        path = self._register(filename, "table")
        df.to_csv(path, index=False)
        return path

    def save_metrics(self, metrics: Dict[str, Any], filename: str = "metrics.json") -> Path:
        """Saves a scalar/nested-dict metrics object to JSON."""
        path = self._register(filename, "metrics")
        with open(path, "w") as f:
            json.dump(metrics, f, indent=2, default=_json_default)
        return path

    def save_figure(self, fig: Figure, filename: str) -> Path:
        """Saves a `matplotlib.figure.Figure` to PNG (`filename` should end in `.png`)."""
        path = self._register(filename, "figure")
        fig.savefig(path, dpi=150, bbox_inches="tight")
        return path

    def save_latex_table(self, df: pd.DataFrame, filename: str, caption: str | None = None,
                          label: str | None = None) -> Path:
        """
        Saves a `pd.DataFrame` as a LaTeX table (`filename` should end in
        `.tex`), via `latex_export.dataframe_to_latex` -- lets a written
        report cite this exact experiment run's tables directly with
        `\\input{...}`/`\\include{...}`, with no manual retyping step
        between notebook and document.
        """
        from .latex_export import dataframe_to_latex  # local import: keeps this an optional artifact

        path = self._register(filename, "latex_table")
        latex_source = dataframe_to_latex(df, caption=caption, label=label)
        with open(path, "w") as f:
            f.write(latex_source)
        return path

    def finalize(self) -> Path:
        """Writes `manifest.json`, listing every artifact produced by this run."""
        path = self.dir / "manifest.json"
        with open(path, "w") as f:
            json.dump(self._manifest, f, indent=2, default=_json_default)
        return path


# ---------------------------------------------------------------------------
# Ready-to-use trackers for the three main experiment types
# ---------------------------------------------------------------------------

def track_pareto_sweep_experiment(name: str, results_df: pd.DataFrame, baseline_metrics: dict,
                                   sim_cfg: Any, train_cfg: Any, quantum_cfg: Any, sweep_cfg: Any,
                                   device: Any, base_dir: str = "experiments") -> ExperimentRun:
    """
    Records one `pareto_sweep.run_pareto_sweep` execution: every config
    dataclass used, the resulting Pareto-frontier table, the baseline
    metrics, and a Pareto-frontier plot (QPU Yield and MAE vs. lambda).
    """
    from . import plotting  # local import: keeps matplotlib optional for callers who never plot

    exp = ExperimentRun(name, base_dir=base_dir)
    exp.save_config({
        "experiment_kind": "pareto_sweep",
        "device": str(device),
        "sim_config": sim_cfg, "train_config": train_cfg,
        "quantum_config": quantum_cfg, "sweep_config": sweep_cfg,
    })
    exp.save_table(results_df, "pareto_frontier.csv")
    exp.save_metrics(baseline_metrics, "baseline_metrics.json")
    exp.save_latex_table(results_df, "pareto_frontier.tex",
                          caption="Pareto Frontier over lambda\\_penalty (mean $\\pm$ std across seeds)",
                          label="tab:pareto_frontier")
    fig = plotting.plot_pareto_frontier(results_df)
    exp.save_figure(fig, "pareto_frontier.png")
    exp.finalize()
    return exp


def track_model_comparison_experiment(name: str, results_df: pd.DataFrame, baseline_metrics: dict,
                                       decision_matrix_df: pd.DataFrame, sensitivity_results: tuple,
                                       sim_cfg: Any, train_cfg: Any, quantum_cfg: Any, baseline_cfg: Any,
                                       comparison_cfg: Any, device: Any,
                                       base_dir: str = "experiments") -> ExperimentRun:
    """
    Records one `model_comparison.run_model_comparison` execution: every
    config dataclass used, the per-model results table, the decision
    matrix, the sensitivity-analysis summary, and three plots (QPU Yield
    bar chart, decision-matrix ranking, sensitivity win-rate).
    """
    from . import plotting

    sensitivity_summary_df, sensitivity_trials_df, sensitivity_verdict = sensitivity_results

    exp = ExperimentRun(name, base_dir=base_dir)
    exp.save_config({
        "experiment_kind": "model_comparison",
        "device": str(device),
        "sim_config": sim_cfg, "train_config": train_cfg, "quantum_config": quantum_cfg,
        "baseline_config": baseline_cfg, "comparison_config": comparison_cfg,
    })
    exp.save_table(results_df, "model_comparison.csv")
    exp.save_table(decision_matrix_df, "decision_matrix.csv")
    exp.save_table(sensitivity_summary_df, "sensitivity_summary.csv")
    exp.save_table(sensitivity_trials_df, "sensitivity_trials.csv")
    exp.save_metrics({**baseline_metrics, "sensitivity_verdict": sensitivity_verdict}, "baseline_metrics.json")
    exp.save_latex_table(results_df, "model_comparison.tex",
                          caption="Cross-architecture model comparison (mean $\\pm$ std across seeds)",
                          label="tab:model_comparison")
    exp.save_latex_table(decision_matrix_df, "decision_matrix.tex",
                          caption="Multi-criteria decision matrix ranking",
                          label="tab:decision_matrix")

    exp.save_figure(plotting.plot_model_comparison_bars(results_df), "qpu_yield_by_model.png")
    exp.save_figure(plotting.plot_decision_matrix(decision_matrix_df), "decision_matrix.png")
    exp.save_figure(plotting.plot_sensitivity_summary(sensitivity_summary_df), "sensitivity_summary.png")
    exp.finalize()
    return exp


def track_ablation_experiment(name: str, results_df: pd.DataFrame, decomposition_df: pd.DataFrame,
                               baseline_metrics: dict, sim_cfg: Any, train_cfg: Any, quantum_cfg: Any,
                               ablation_cfg: Any, device: Any,
                               base_dir: str = "experiments") -> ExperimentRun:
    """
    Records one `ablation.run_ablation_study` execution: every config
    dataclass used, the per-cell results table, the 2x2 factorial
    decomposition table, a per-cell QPU-Yield bar chart, and one
    architecture x loss interaction plot per headline metric in
    `ablation_cfg.headline_metrics`.
    """
    from . import plotting

    exp = ExperimentRun(name, base_dir=base_dir)
    exp.save_config({
        "experiment_kind": "ablation",
        "device": str(device),
        "sim_config": sim_cfg, "train_config": train_cfg,
        "quantum_config": quantum_cfg, "ablation_config": ablation_cfg,
    })
    exp.save_table(results_df, "ablation_results.csv")
    exp.save_table(decomposition_df, "ablation_decomposition.csv")
    exp.save_metrics(baseline_metrics, "baseline_metrics.json")
    exp.save_latex_table(results_df, "ablation_results.tex",
                          caption="2x2 factorial ablation results (mean $\\pm$ std across seeds)",
                          label="tab:ablation_results")
    exp.save_latex_table(decomposition_df.drop(columns=["Interpretation"], errors="ignore"),
                          "ablation_decomposition.tex",
                          caption="Architecture / Loss / Interaction effect decomposition",
                          label="tab:ablation_decomposition")

    exp.save_figure(plotting.plot_model_comparison_bars(results_df, metric_col="QPU Yield (%)",
                                                          model_col="Model"), "qpu_yield_by_cell.png")
    for metric in decomposition_df["Metric"]:
        fig = plotting.plot_ablation_interaction(decomposition_df, metric)
        exp.save_figure(fig, f"interaction_{metric}.png")
    exp.finalize()
    return exp
