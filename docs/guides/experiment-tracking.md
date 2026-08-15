# Experiment tracking

Quantum Twin has two tracking backends, usable independently or together.

## Local (always on, zero setup)

`quantum_twin.experiment_tracking.ExperimentRun` writes every run's
configuration, result tables, and figures to a timestamped directory --
no external service required, works offline. This is the DEFAULT and
requires no configuration:

```python
from quantum_twin.experiment_tracking import track_pareto_sweep_experiment

exp = track_pareto_sweep_experiment(
    "pareto_sweep_v1", results_df, baseline_metrics,
    sim_cfg, train_cfg, quantum_cfg, sweep_cfg, device,
)
```

Produces, under `experiments/pareto_sweep_v1_<timestamp>/`:

```
config.json           # every dataclass config used, flattened to JSON
metrics.json           # scalar metrics not already inside a table
pareto_frontier.csv     # the results table
pareto_frontier.tex      # the same table, rendered as a LaTeX \input{}-ready file
pareto_frontier.png       # the Pareto Frontier plot
manifest.json              # lists every file above
```

## MLflow (optional)

For teams that already run an MLflow tracking server, or want a
searchable run history and side-by-side comparison UI,
`quantum_twin.mlops.MLflowTracker` mirrors the same small API surface
(`log_config`/`log_table`/`log_metrics`/`log_figure`) against MLflow's
tracking API:

```python
from quantum_twin.mlops import MLflowTracker

with MLflowTracker("pareto_sweep", run_name="lambda_sweep_v1") as tracker:
    tracker.log_config({"sim_config": sim_cfg, "train_config": train_cfg})
    tracker.log_table(results_df, "pareto_frontier")
    tracker.log_metrics({"best_qpu_yield_pct": 92.3})
    tracker.log_figure(fig, "pareto_frontier")
```

Install with `pip install -e ".[mlflow]"` to enable it.
**`MLflowTracker` degrades gracefully when `mlflow` isn't installed**:
every method becomes a safe no-op (with a single warning printed once
per process, not once per call) -- calling this in an environment
without MLflow never raises and never affects the local `ExperimentRun`
artifacts, which remain the tracking source of truth either way.

```python
from quantum_twin.mlops import MLFLOW_AVAILABLE

if not MLFLOW_AVAILABLE:
    print("MLflow tracking is disabled; local ExperimentRun artifacts are unaffected.")
```
