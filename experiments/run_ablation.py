#!/usr/bin/env python3
"""
Runs the 2x2 factorial ablation study ({EdgeLSTM, StandardLSTM} x
{MSE, CS-MSE}) and records the architecture/loss/interaction effect
decomposition with `quantum_twin.experiment_tracking.track_ablation_experiment`.

Usage
-----
    python experiments/run_ablation.py --epochs 150 --representative-lambda 10.0
"""

from __future__ import annotations

import argparse

from quantum_twin.ablation import run_ablation_study
from quantum_twin.channel_simulator import WDMChannelSimulator
from quantum_twin.cli import get_device
from quantum_twin.config import AblationConfig, QuantumConfig, SimConfig, TrainConfig
from quantum_twin.experiment_tracking import track_ablation_experiment
from quantum_twin.reproducibility import set_full_determinism


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(range(42, 52)))
    parser.add_argument("--representative-lambda", type=float, default=10.0)
    parser.add_argument("--experiment-name", type=str, default="ablation_study")
    parser.add_argument("--experiments-dir", type=str, default="experiments_output")
    parser.add_argument("--track-mlflow", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = get_device()
    set_full_determinism(seed=args.seeds[0])

    sim_cfg = SimConfig()
    train_cfg = TrainConfig(epochs=args.epochs)
    quantum_cfg = QuantumConfig()
    ablation_cfg = AblationConfig(
        epochs=args.epochs, representative_lambda=args.representative_lambda, seeds=args.seeds,
    )

    wdm_sim = WDMChannelSimulator(n_steps=sim_cfg.n_steps, dt=sim_cfg.dt, seed=sim_cfg.seed)
    df = wdm_sim.generate_dataset()
    X_train, y_train, X_test, y_test, _scaler = wdm_sim.preprocess(
        df, window_size=sim_cfg.window_size, test_size=sim_cfg.test_size,
    )
    X_train, y_train = X_train.to(device), y_train.to(device)
    X_test, y_test = X_test.to(device), y_test.to(device)

    results_df, decomposition_df, baseline_metrics, _per_cell_seed_results = run_ablation_study(
        X_train, y_train, X_test, y_test, device=device,
        train_cfg=train_cfg, quantum_cfg=quantum_cfg, ablation_cfg=ablation_cfg,
    )

    print(results_df.to_string(index=False))
    print("\nFactorial decomposition:\n", decomposition_df.to_string(index=False))
    for _, row in decomposition_df.iterrows():
        print(f"\n[{row['Metric']}] {row['Interpretation']}")

    exp = track_ablation_experiment(
        args.experiment_name, results_df, decomposition_df, baseline_metrics,
        sim_cfg, train_cfg, quantum_cfg, ablation_cfg, device,
        base_dir=args.experiments_dir,
    )
    print(f"\nLocal artifacts saved to: {exp.dir}")

    if args.track_mlflow:
        from quantum_twin.mlops import MLflowTracker

        with MLflowTracker(args.experiment_name) as tracker:
            tracker.log_config({
                "sim_config": sim_cfg, "train_config": train_cfg,
                "quantum_config": quantum_cfg, "ablation_config": ablation_cfg,
            })
            tracker.log_table(results_df, "ablation_results")
            tracker.log_table(decomposition_df, "ablation_decomposition")


if __name__ == "__main__":
    main()
