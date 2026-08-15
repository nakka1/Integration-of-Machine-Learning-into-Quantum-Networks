# Running experiments

Every experiment type has both a library function (for use inside a
notebook or another script) and a standalone script under `experiments/`
(for `python experiments/run_*.py` from the command line).

| Experiment | Library function | Script |
|---|---|---|
| Pareto sweep over `lambda_penalty` | `quantum_twin.pareto_sweep.run_pareto_sweep` | `experiments/run_pareto_sweep.py` |
| Cross-architecture comparison | `quantum_twin.model_comparison.run_model_comparison` | `experiments/run_model_comparison.py` |
| 2x2 factorial ablation | `quantum_twin.ablation.run_ablation_study` | `experiments/run_ablation.py` |
| Walk-forward cross-validation | `quantum_twin.walk_forward.run_walk_forward_evaluation` | `experiments/run_walk_forward.py` |

All four scripts accept `--track-mlflow` (except `run_walk_forward.py`,
which is a validation pass rather than a trackable experiment) to also
record the run via `quantum_twin.mlops.MLflowTracker`, on top of the
always-on local `ExperimentRun` artifacts.

## Example: the full CLI pipeline

`quantum_twin.cli.main` (or the `quantum-twin` console script) chains the
Pareto sweep with the optional baseline comparison and ablation study in
one call:

```bash
quantum-twin \
    --epochs 150 \
    --lambda-values 1 2 5 10 20 50 \
    --compare-baselines --representative-lambda 10.0 \
    --run-ablation
```

```python
from quantum_twin.cli import main

results = main(run_baseline_comparison=True, run_ablation=True)
results_df, baseline_metrics, per_seed_results, comparison_results, ablation_results = results
```

## Example: one experiment, tracked locally

```python
from quantum_twin.experiment_tracking import track_pareto_sweep_experiment

exp = track_pareto_sweep_experiment(
    "pareto_sweep_v1", results_df, baseline_metrics,
    sim_cfg, train_cfg, quantum_cfg, sweep_cfg, device,
)
print(exp.dir)  # config.json, pareto_frontier.{csv,tex}, pareto_frontier.png, manifest.json
```
