#!/usr/bin/env python3
"""
Runs the cross-architecture model comparison (EdgeLSTM+CS-MSE vs.
LSTM+MSE, Random Forest, XGBoost, Transformer, Persistence,
MovingAverage, Oracle), the ranked decision matrix, the +/-10%
weight-sensitivity analysis, and the paired statistical-significance test
against every baseline -- then records the result with
`quantum_twin.experiment_tracking.track_model_comparison_experiment`.

Usage
-----
    python experiments/run_model_comparison.py --epochs 150 --representative-lambda 10.0
    python experiments/run_model_comparison.py --no-xgboost --track-mlflow
"""

from __future__ import annotations

import argparse

from quantum_twin.channel_simulator import WDMChannelSimulator
from quantum_twin.cli import get_device
from quantum_twin.config import (
    BaselineConfig,
    ComparisonConfig,
    EnergyConfig,
    QuantumConfig,
    SimConfig,
    TrainConfig,
)
from quantum_twin.experiment_tracking import track_model_comparison_experiment
from quantum_twin.model_comparison import run_model_comparison
from quantum_twin.reproducibility import set_full_determinism
from quantum_twin.statistics_tests import compare_models_statistically


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(range(42, 52)))
    parser.add_argument("--representative-lambda", type=float, default=10.0)
    parser.add_argument("--no-xgboost", action="store_true")
    parser.add_argument("--sensitivity-pct", type=float, default=0.10)
    parser.add_argument("--experiment-name", type=str, default="model_comparison")
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
    baseline_cfg = BaselineConfig()
    energy_cfg = EnergyConfig()
    comparison_cfg = ComparisonConfig(
        representative_lambda=args.representative_lambda, seeds=args.seeds,
        include_xgboost=not args.no_xgboost, sensitivity_perturbation_pct=args.sensitivity_pct,
    )

    wdm_sim = WDMChannelSimulator(n_steps=sim_cfg.n_steps, dt=sim_cfg.dt, seed=sim_cfg.seed)
    df = wdm_sim.generate_dataset()
    X_train, y_train, X_test, y_test, _scaler = wdm_sim.preprocess(
        df, window_size=sim_cfg.window_size, test_size=sim_cfg.test_size,
    )
    X_train, y_train = X_train.to(device), y_train.to(device)
    X_test, y_test = X_test.to(device), y_test.to(device)

    (results_df, baseline_metrics, decision_matrix_df,
     per_model_seed_results, sensitivity_results) = run_model_comparison(
        X_train, y_train, X_test, y_test, device=device,
        train_cfg=train_cfg, quantum_cfg=quantum_cfg, baseline_cfg=baseline_cfg,
        energy_cfg=energy_cfg, comparison_cfg=comparison_cfg,
    )

    print(results_df.to_string(index=False))
    print("\nDecision matrix:\n", decision_matrix_df.to_string(index=False))

    significance_df = compare_models_statistically(
        per_model_seed_results, metric_key="qpu_yield_pct",
        reference_model="EdgeLSTM+CS-MSE",
    )
    print("\nStatistical significance vs. EdgeLSTM+CS-MSE (QPU Yield):\n",
          significance_df.to_string(index=False))

    exp = track_model_comparison_experiment(
        args.experiment_name, results_df, baseline_metrics, decision_matrix_df,
        sensitivity_results, sim_cfg, train_cfg, quantum_cfg, baseline_cfg, comparison_cfg, device,
        base_dir=args.experiments_dir,
    )
    exp.save_table(significance_df, "significance_vs_edgelstm.csv")
    exp.finalize()
    print(f"\nLocal artifacts saved to: {exp.dir}")

    if args.track_mlflow:
        from quantum_twin.mlops import MLflowTracker

        with MLflowTracker(args.experiment_name) as tracker:
            tracker.log_config({
                "sim_config": sim_cfg, "train_config": train_cfg, "quantum_config": quantum_cfg,
                "baseline_config": baseline_cfg, "comparison_config": comparison_cfg,
            })
            tracker.log_table(results_df, "model_comparison")
            tracker.log_table(decision_matrix_df, "decision_matrix")
            tracker.log_table(significance_df, "significance")


if __name__ == "__main__":
    main()
