#!/usr/bin/env python3
"""
Runs the Pareto Frontier sweep over `lambda_penalty` and records the
result with `quantum_twin.experiment_tracking.track_pareto_sweep_experiment`
(local CSV/PNG/TeX/JSON artifacts) and, if the optional `mlflow` extra is
installed, also with `quantum_twin.mlops.MLflowTracker`.

This script lives OUTSIDE `src/quantum_twin/` on purpose: it is a
*consumer* of the library (imports it exactly the way an external user
would, via `pip install -e .` / `pip install quantum-twin`), not part of
the library's own source tree -- separation of concerns between
"reusable, importable code" (`src/quantum_twin/`) and "one specific
experiment run" (this file).

Usage
-----
    python experiments/run_pareto_sweep.py --epochs 150 --seeds 42 43 44
    python experiments/run_pareto_sweep.py --lambda-values 1 2 5 10 20 50 --track-mlflow
"""

from __future__ import annotations

import argparse

from quantum_twin.channel_simulator import WDMChannelSimulator
from quantum_twin.cli import get_device
from quantum_twin.config import QuantumConfig, SimConfig, SweepConfig, TrainConfig
from quantum_twin.experiment_tracking import track_pareto_sweep_experiment
from quantum_twin.pareto_sweep import run_pareto_sweep
from quantum_twin.reproducibility import set_full_determinism


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--lr", type=float, default=0.012)
    parser.add_argument("--hidden-size", type=int, default=16)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(range(42, 52)))
    parser.add_argument("--lambda-values", type=float, nargs="+",
                         default=[1.0, 2.0, 5.0, 10.0, 20.0, 50.0])
    parser.add_argument("--experiment-name", type=str, default="pareto_sweep")
    parser.add_argument("--experiments-dir", type=str, default="experiments_output")
    parser.add_argument("--track-mlflow", action="store_true",
                         help="Also record this run with MLflow (requires `pip install "
                              "\"quantum-twin[mlflow]\"`; no-ops with a warning otherwise).")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = get_device()
    set_full_determinism(seed=args.seeds[0])

    sim_cfg = SimConfig()
    train_cfg = TrainConfig(epochs=args.epochs, lr=args.lr, hidden_size=args.hidden_size)
    quantum_cfg = QuantumConfig()
    sweep_cfg = SweepConfig(lambda_values=args.lambda_values, seeds=args.seeds)

    wdm_sim = WDMChannelSimulator(n_steps=sim_cfg.n_steps, dt=sim_cfg.dt, seed=sim_cfg.seed)
    df = wdm_sim.generate_dataset()
    X_train, y_train, X_test, y_test, _scaler = wdm_sim.preprocess(
        df, window_size=sim_cfg.window_size, test_size=sim_cfg.test_size,
    )
    X_train, y_train = X_train.to(device), y_train.to(device)
    X_test, y_test = X_test.to(device), y_test.to(device)

    results_df, baseline_metrics, _per_seed_results = run_pareto_sweep(
        sweep_cfg.lambda_values, X_train, y_train, X_test, y_test, device=device,
        threshold=train_cfg.threshold, epochs=train_cfg.epochs, lr=train_cfg.lr,
        hidden_size=train_cfg.hidden_size,
        T1=quantum_cfg.T1, T2=quantum_cfg.T2, depol_prob=quantum_cfg.depol_prob,
        shots=quantum_cfg.shots, quantum_seed=quantum_cfg.seed,
        lambda_fn=train_cfg.lambda_fn, discard_penalty_weight=train_cfg.discard_penalty_weight,
        max_discard_rate=train_cfg.max_discard_rate, seeds=sweep_cfg.seeds,
    )

    print(results_df.to_string(index=False))

    exp = track_pareto_sweep_experiment(
        args.experiment_name, results_df, baseline_metrics,
        sim_cfg, train_cfg, quantum_cfg, sweep_cfg, device,
        base_dir=args.experiments_dir,
    )
    print(f"\nLocal artifacts saved to: {exp.dir}")

    if args.track_mlflow:
        from quantum_twin.mlops import MLflowTracker

        with MLflowTracker(args.experiment_name) as tracker:
            tracker.log_config({
                "sim_config": sim_cfg, "train_config": train_cfg,
                "quantum_config": quantum_cfg, "sweep_config": sweep_cfg,
            })
            tracker.log_table(results_df, "pareto_frontier")
            tracker.log_metrics({k: v for k, v in baseline_metrics.items() if isinstance(v, (int, float))})


if __name__ == "__main__":
    main()
