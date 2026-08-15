"""
Command-line entry point.

Previously, `main()` lived inside the notebook and could only be run cell
by cell, inside a Jupyter kernel. Here it is an ordinary, importable
library function (`from quantum_twin.cli import main`) and is also
runnable via `python -m quantum_twin.cli` from a plain terminal --
no notebook required.
"""

from __future__ import annotations

import argparse
from typing import Any

import torch

from .channel_simulator import WDMChannelSimulator
from .config import (
    AblationConfig,
    BaselineConfig,
    ComparisonConfig,
    EnergyConfig,
    QuantumConfig,
    SimConfig,
    SweepConfig,
    TrainConfig,
)
from .pareto_sweep import run_pareto_sweep
from .model_comparison import run_model_comparison
from .ablation import run_ablation_study
from .reproducibility import set_full_determinism


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main(sim_cfg: SimConfig | None = None, train_cfg: TrainConfig | None = None,
          quantum_cfg: QuantumConfig | None = None, sweep_cfg: SweepConfig | None = None,
          device: torch.device | None = None, base_seed: int = 42,
          run_baseline_comparison: bool = False,
          baseline_cfg: BaselineConfig | None = None, energy_cfg: EnergyConfig | None = None,
          comparison_cfg: ComparisonConfig | None = None,
          run_ablation: bool = False, ablation_cfg: AblationConfig | None = None,
          deterministic_algorithms: bool = True) -> tuple[Any, ...]:
    """
    Full pipeline of the Quantum Repeater Digital Twin (v3.2, decomposed).

    1. Generates and preprocesses the synthetic dataset (WDMChannelSimulator).
    2. Moves tensors to the selected device (CPU/GPU).
    3. Runs the (multi-seed) Pareto Frontier sweep over lambda_penalty,
       including the one-time computation of the blind/reactive baseline.
    4. Prints the consolidated metrics table.
    5. (Optional, `run_baseline_comparison=True`) Runs the cross-architecture
       baseline comparison (LSTM+MSE, Random Forest, XGBoost, Transformer)
       with regression accuracy (MAE/RMSE/R^2), the admission confusion
       matrix (FP/FN/TP/TN), throughput/QPU-economy/energy accounting, the
       dimensionless C_latencia = tau_inf/T2 ratio, the ranked decision
       matrix, and an automatic +/-10% decision-weight sensitivity
       analysis.
    6. (Optional, `run_ablation=True`) Runs the 2x2 factorial ablation
       study ({EdgeLSTM, StandardLSTM} x {MSE, CS-MSE}), reporting the
       architecture/loss/interaction effect decomposition.

    `deterministic_algorithms` (default `True`) additionally forces
    PyTorch's CUDA/cuDNN backend into its deterministic algorithm variants
    via `reproducibility.set_full_determinism` -- see that function's
    docstring for the exact flags set and their (usually modest) run-time
    cost; set to `False` to skip this and keep only the per-seed
    reproducibility every training round already gets from
    `seed_everything`.

    Returns a variable-length tuple whose SHAPE depends on which optional
    stages ran (precise `-> tuple[Any, ...]` typing is intentional here --
    the alternative, encoding all four possible return shapes as
    `@overload`s keyed on two independent boolean flags, would be four
    times the type-checking machinery for one CLI-glue function):
        - always                              : `(results_df, baseline_metrics, per_seed_results)`
        - `run_baseline_comparison=True` only  : `(..., comparison_results)`
        - `run_ablation=True` only              : `(..., ablation_results)`  (comparison_results is `None`)
        - both `True`                            : `(..., comparison_results, ablation_results)`
    """
    sim_cfg = sim_cfg or SimConfig()
    train_cfg = train_cfg or TrainConfig()
    quantum_cfg = quantum_cfg or QuantumConfig()
    sweep_cfg = sweep_cfg or SweepConfig()
    baseline_cfg = baseline_cfg or BaselineConfig()
    energy_cfg = energy_cfg or EnergyConfig()
    comparison_cfg = comparison_cfg or ComparisonConfig()
    ablation_cfg = ablation_cfg or AblationConfig()
    device = device or get_device()

    # Process-wide determinism: seeds Python/NumPy/Torch(CPU+CUDA) and
    # forces PyTorch's deterministic algorithm variants. Must run before
    # any tensor touches the GPU (see reproducibility.set_full_determinism).
    set_full_determinism(base_seed, deterministic_algorithms=deterministic_algorithms)

    print("=" * 88)
    print(" QUANTUM REPEATER DIGITAL TWIN -- PARETO FRONTIER (CS_MSELoss) ".center(88, "="))
    print("=" * 88)
    print(f"\nDevice: {device}")

    # -----------------------------------------------------------------
    # 1) Data generation and preprocessing
    # -----------------------------------------------------------------
    print("\n[1/3] Generating synthetic dataset (Ornstein-Uhlenbeck) ...")
    wdm_sim = WDMChannelSimulator(n_steps=sim_cfg.n_steps, dt=sim_cfg.dt, seed=sim_cfg.seed)
    df = wdm_sim.generate_dataset()
    X_train, y_train, X_test, y_test, feat_scaler = wdm_sim.preprocess(
        df, window_size=sim_cfg.window_size, test_size=sim_cfg.test_size,
    )
    print(f"    Total samples: {len(df)} | Training windows: {len(X_train)} | Test windows: {len(X_test)}")
    print(f"    Fraction of true fidelity below threshold {train_cfg.threshold}: "
          f"{(df['fidelity'] < train_cfg.threshold).mean() * 100:.1f}%")

    # -----------------------------------------------------------------
    # 2) Device handling: moves tensors to the selected device
    # -----------------------------------------------------------------
    X_train, y_train = X_train.to(device), y_train.to(device)
    X_test, y_test = X_test.to(device), y_test.to(device)

    # -----------------------------------------------------------------
    # 3) Pareto Frontier sweep over lambda_penalty (multi-seed averaging)
    # -----------------------------------------------------------------
    print(f"\n[2/3] Running the Pareto Frontier for lambda_penalty = "
          f"{sweep_cfg.lambda_values} with {len(sweep_cfg.seeds)} seeds/point "
          f"({sweep_cfg.seeds}) ...\n")

    results_df, baseline_metrics, per_seed_results = run_pareto_sweep(
        sweep_cfg.lambda_values, X_train, y_train, X_test, y_test, device=device,
        threshold=train_cfg.threshold, epochs=train_cfg.epochs, lr=train_cfg.lr,
        hidden_size=train_cfg.hidden_size,
        T1=quantum_cfg.T1, T2=quantum_cfg.T2, depol_prob=quantum_cfg.depol_prob,
        shots=quantum_cfg.shots, quantum_seed=quantum_cfg.seed,
        lambda_fn=train_cfg.lambda_fn, discard_penalty_weight=train_cfg.discard_penalty_weight,
        max_discard_rate=train_cfg.max_discard_rate, seeds=sweep_cfg.seeds,
    )

    # -----------------------------------------------------------------
    # Final report
    # -----------------------------------------------------------------
    print("[3/3] Consolidating results ...\n")
    print("=" * 88)
    print(" BASELINE (Blind/Reactive Purification) ".center(88, "="))
    print("=" * 88)
    print(f"  Total cycles evaluated    : {baseline_metrics['total_steps']}")
    print(f"  Purification attempts     : {baseline_metrics['attempted']} (unconditional admission)")
    print(f"  Useful pairs obtained     : {baseline_metrics['useful_pairs']}")
    print(f"  Forced classical latency  : {baseline_metrics['avg_classical_latency_s']*1000:.4f} ms "
          f"(neural network never invoked)")
    print(f"  Confusion (degenerate)    : TP={baseline_metrics['confusion_matrix']['TP']} "
          f"FP={baseline_metrics['confusion_matrix']['FP']} "
          f"(every dead photon is admitted, by construction -- FN=TN=0)")

    print("\n" + "=" * 88)
    print(" PARETO FRONTIER: CS_MSELoss(lambda_penalty) -- mean +/- std (multi-seed) "
          .center(88, "="))
    print("=" * 88)
    print(results_df.to_string(index=False))
    print("=" * 88)

    comparison_results = None
    if run_baseline_comparison:
        # -----------------------------------------------------------------
        # 4) (Optional) Cross-architecture baseline comparison
        # -----------------------------------------------------------------
        print("\n[4/5] Running cross-architecture baseline comparison "
              f"(LSTM+MSE, Random Forest, XGBoost, Transformer) with "
              f"{len(comparison_cfg.seeds)} seeds/model ...\n")

        comp_results_df, comp_baseline_metrics, decision_matrix_df, per_model_seed_results, sensitivity_results = run_model_comparison(
            X_train, y_train, X_test, y_test, device=device,
            train_cfg=train_cfg, quantum_cfg=quantum_cfg, baseline_cfg=baseline_cfg,
            energy_cfg=energy_cfg, comparison_cfg=comparison_cfg,
        )

        print("\n" + "=" * 88)
        print(" BASELINE COMPARISON: regression accuracy / confusion / throughput / QPU economy / energy "
              .center(88, "="))
        print("=" * 88)
        print(comp_results_df.to_string(index=False))

        print("\n" + "=" * 88)
        print(" DECISION MATRIX (weighted, higher Decision Score = better; Rank 1 = recommended) "
              .center(88, "="))
        print("=" * 88)
        print(decision_matrix_df.to_string(index=False))
        print("=" * 88)

        sensitivity_summary_df, sensitivity_trials_df, sensitivity_verdict = sensitivity_results
        print("\n" + "=" * 88)
        print(f" DECISION-WEIGHT SENSITIVITY (+/-{comparison_cfg.sensitivity_perturbation_pct*100:.0f}%, "
              f"{2**len(comparison_cfg.decision_weights)} vertex combinations) ".center(88, "="))
        print("=" * 88)
        print(sensitivity_summary_df.to_string(index=False))
        print(f"\n  Verdict: {sensitivity_verdict}")
        print("=" * 88)

        comparison_results = (comp_results_df, comp_baseline_metrics, decision_matrix_df,
                               per_model_seed_results, sensitivity_results)

    ablation_results = None
    if run_ablation:
        # -----------------------------------------------------------------
        # 5) (Optional) 2x2 factorial ablation study
        # -----------------------------------------------------------------
        print(f"\n[5/5] Running 2x2 factorial ablation study "
              f"({{EdgeLSTM, StandardLSTM}} x {{MSE, CS-MSE}}) with "
              f"{len(ablation_cfg.seeds)} seeds/cell ...\n")

        ablation_results_df, decomposition_df, ablation_baseline_metrics, per_cell_seed_results = run_ablation_study(
            X_train, y_train, X_test, y_test, device=device,
            train_cfg=train_cfg, quantum_cfg=quantum_cfg, ablation_cfg=ablation_cfg,
            energy_cfg=energy_cfg, comparison_cfg=comparison_cfg,
        )

        print("\n" + "=" * 88)
        print(" ABLATION STUDY: {EdgeLSTM, StandardLSTM} x {MSE, CS-MSE} ".center(88, "="))
        print("=" * 88)
        print(ablation_results_df.to_string(index=False))

        print("\n" + "=" * 88)
        print(" ABLATION DECOMPOSITION: architecture effect / loss effect / interaction ".center(88, "="))
        print("=" * 88)
        print(decomposition_df[["Metric", "Architecture Effect", "Loss Effect", "Interaction Effect"]].to_string(index=False))
        for _, row in decomposition_df.iterrows():
            print(f"\n  [{row['Metric']}] {row['Interpretation']}")
        print("=" * 88)

        ablation_results = (ablation_results_df, decomposition_df, ablation_baseline_metrics, per_cell_seed_results)

    if not run_baseline_comparison and not run_ablation:
        return results_df, baseline_metrics, per_seed_results
    if not run_ablation:
        return results_df, baseline_metrics, per_seed_results, comparison_results
    return results_df, baseline_metrics, per_seed_results, comparison_results, ablation_results


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Quantum Repeater Digital Twin -- Pareto Frontier")
    parser.add_argument("--epochs", type=int, default=150, help="Training epochs per round (seed x lambda).")
    parser.add_argument("--lr", type=float, default=0.012, help="Adam learning rate.")
    parser.add_argument("--hidden-size", type=int, default=16, help="EdgeLSTM hidden state size.")
    parser.add_argument("--seeds", type=int, nargs="+", default=list(range(42, 52)),
                         help="Seeds used in multi-seed averaging (one training round per seed x lambda).")
    parser.add_argument("--lambda-values", type=float, nargs="+", default=[1.0, 2.0, 5.0, 10.0, 20.0, 50.0],
                         help="lambda_penalty values swept over the Pareto Frontier.")
    parser.add_argument("--compare-baselines", action="store_true",
                         help="Also run the cross-architecture baseline comparison "
                              "(LSTM+MSE, Random Forest, XGBoost, Transformer) with "
                              "throughput/QPU-economy/energy metrics and the decision matrix.")
    parser.add_argument("--no-xgboost", action="store_true",
                         help="Skip the XGBoost baseline even if the 'xgboost' package is installed.")
    parser.add_argument("--representative-lambda", type=float, default=10.0,
                         help="lambda_penalty used for the EdgeLSTM+CS-MSE row in --compare-baselines.")
    parser.add_argument("--sensitivity-pct", type=float, default=0.10,
                         help="+/- fractional perturbation applied to each decision-matrix weight "
                              "in the automatic weight-sensitivity analysis (default 0.10 = +/-10%%).")
    parser.add_argument("--run-ablation", action="store_true",
                         help="Also run the 2x2 factorial ablation study "
                              "({EdgeLSTM, StandardLSTM} x {MSE, CS-MSE}), reporting the "
                              "architecture/loss/interaction effect decomposition.")
    return parser


def cli_entrypoint() -> None:
    """
    Console-script entry point (see `pyproject.toml`'s `[project.scripts]`):
    installing this package (`pip install -e .`) puts a `quantum-twin`
    command on `PATH` that runs exactly this function, equivalent to
    `python -m quantum_twin.cli`. Kept as a thin wrapper around the same
    `build_arg_parser()` / `main()` pair the `__main__` block below uses,
    so there is exactly one place that translates CLI flags into config
    dataclasses.
    """
    args = build_arg_parser().parse_args()
    main(
        train_cfg=TrainConfig(epochs=args.epochs, lr=args.lr, hidden_size=args.hidden_size),
        sweep_cfg=SweepConfig(lambda_values=args.lambda_values, seeds=args.seeds),
        run_baseline_comparison=args.compare_baselines,
        comparison_cfg=ComparisonConfig(
            representative_lambda=args.representative_lambda,
            seeds=args.seeds,
            include_xgboost=not args.no_xgboost,
            sensitivity_perturbation_pct=args.sensitivity_pct,
        ),
        run_ablation=args.run_ablation,
        ablation_cfg=AblationConfig(seeds=args.seeds, representative_lambda=args.representative_lambda),
    )


if __name__ == "__main__":
    cli_entrypoint()
