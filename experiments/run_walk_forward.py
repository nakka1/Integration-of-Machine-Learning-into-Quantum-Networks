#!/usr/bin/env python3
"""
Runs walk-forward (rolling-origin) temporal cross-validation for
EdgeLSTM+CS-MSE, validating that the reported gains hold up across
several independent, non-overlapping, forward-moving slices of time
rather than resting on a single chronological train/test split.

Usage
-----
    python experiments/run_walk_forward.py --n-splits 5 --test-size 150 --min-train-size 300
"""

from __future__ import annotations

import argparse

from quantum_twin.channel_simulator import WDMChannelSimulator
from quantum_twin.cli import get_device
from quantum_twin.config import QuantumConfig, SimConfig, TrainConfig, WalkForwardConfig
from quantum_twin.reproducibility import set_full_determinism
from quantum_twin.walk_forward import run_walk_forward_evaluation


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--test-size", type=int, default=150)
    parser.add_argument("--min-train-size", type=int, default=300)
    parser.add_argument("--rolling", action="store_true",
                         help="Use a fixed-size rolling window instead of an expanding one.")
    parser.add_argument("--representative-lambda", type=float, default=10.0)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = get_device()
    set_full_determinism(seed=args.seed)

    sim_cfg = SimConfig()
    train_cfg = TrainConfig(epochs=args.epochs)
    quantum_cfg = QuantumConfig()
    wf_cfg = WalkForwardConfig(
        n_splits=args.n_splits, test_size=args.test_size, min_train_size=args.min_train_size,
        expanding=not args.rolling, epochs=args.epochs,
        representative_lambda=args.representative_lambda, seed=args.seed,
    )

    wdm_sim = WDMChannelSimulator(n_steps=sim_cfg.n_steps, dt=sim_cfg.dt, seed=sim_cfg.seed)
    df = wdm_sim.generate_dataset()
    # test_size=0.0: every windowed sample lands in the "train" output;
    # run_walk_forward_evaluation performs its own chronological splitting.
    X_full, y_full, _unused_X, _unused_y, _scaler = wdm_sim.preprocess(
        df, window_size=sim_cfg.window_size, test_size=0.0,
    )

    fold_df, summary_df, _splits = run_walk_forward_evaluation(
        X_full, y_full, device, train_cfg=train_cfg, quantum_cfg=quantum_cfg, wf_cfg=wf_cfg,
    )

    print(fold_df.to_string(index=False))
    print("\nAggregate summary (mean + 95% CI across folds):\n", summary_df.to_string(index=False))


if __name__ == "__main__":
    main()
