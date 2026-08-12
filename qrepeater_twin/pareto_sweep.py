"""
Pareto Frontier -- `lambda_penalty` sweep with multi-seed averaging.

Fixes the "Statistical Fragility" flaw: the original prototype trained the
EdgeLSTM with a single batch (full-batch) and a single random seed per
lambda value, merely *acknowledging* in markdown that individual points
could get stuck in local optima -- without actually mitigating the problem.

Here, each `lambda_penalty` value is trained and evaluated `len(seeds)`
times independently (a different seed each round, applied before the
EdgeLSTM's weight initialization). The point reported on the Pareto
Frontier is the mean across those rounds, accompanied by the standard
deviation -- which makes the training variance at each lambda visible
instead of hidden, drastically reducing the risk of making a production
decision (which lambda to deploy) based on a single, unrepresentative
local optimum.

Each row also reports, per lambda:
    - MAE / RMSE / R^2   : pure regression quality of F_hat(t) itself
                             (`metrics.evaluate_predictor_regression`),
                             computed independently of the admission loop.
    - FP / FN counts       : the admission confusion matrix
                               (`metrics.compute_confusion_metrics`) --
                               the direct, countable evidence for why
                               `CS_MSELoss.lambda_penalty` is worth
                               sweeping: FP ("dead photon admitted") should
                               fall as lambda grows, at some cost in FN
                               ("good photon discarded").
    - C_latencia = tau_inf / T2 : the dimensionless temporal-scale ratio
                                    (`metrics.compute_latency_ratio`)
                                    replacing a raw-millisecond comparison
                                    with a physical constraint against the
                                    qubit's own coherence time.
"""

from __future__ import annotations

import statistics as stats
from typing import List, Sequence

import pandas as pd
import torch

from .models import EdgeLSTM, train_edge_lstm
from .metrics import compute_confusion_metrics, compute_latency_ratio, evaluate_predictor_regression
from .orchestrator import DigitalTwinOrchestrator
from .quantum_node import QuantumRepeaterNode


def _mean_std(values: Sequence[float]) -> tuple:
    mean = stats.fmean(values)
    std = stats.pstdev(values) if len(values) > 1 else 0.0
    return mean, std


def run_pareto_sweep(lambda_values: list, X_train: torch.Tensor, y_train: torch.Tensor,
                      X_test: torch.Tensor, y_test: torch.Tensor, device: torch.device,
                      threshold: float = 0.65, epochs: int = 120, lr: float = 3e-3,
                      hidden_size: int = 16, T1: float = 50e-6, T2: float = 30e-6,
                      depol_prob: float = 0.01, shots: int = 512, quantum_seed: int = 7,
                      lambda_fn: float = 2.0, discard_penalty_weight: float = 5.0,
                      max_discard_rate: float = 0.60, seeds: List[int] = None):
    """
    Runs the Pareto Frontier sweep over the `lambda_penalty` hyperparameter
    of CS_MSELoss, with multi-seed averaging at every point.

    Parameters
    ----------
    seeds : list[int], optional
        Seeds used to repeat training/evaluation at each lambda value.
        Default: 5 seeds ([42, 43, 44, 45, 46]). Each round trains an
        EdgeLSTM from scratch (weight initialization determined by the
        seed) and runs the full Digital Twin over the test set; the
        results of the `len(seeds)` rounds are aggregated into mean ±
        standard deviation before composing that lambda's row in the
        final table.

    Returns
    -------
    results_df : pd.DataFrame
        Consolidated table with one row per lambda value, reporting the
        mean ± standard deviation of each metric across seeds.
    baseline_metrics : dict
        Blind/reactive baseline metrics, computed exactly once (they don't
        depend on lambda or seed, since the neural network is never
        invoked).
    per_seed_results : dict[float, list[dict]]
        Raw metrics from each individual round (lambda -> list of dicts),
        preserved for auditing/debugging and to allow recomputing other
        statistics (median, confidence intervals, etc.) without retraining.
    """
    if seeds is None:
        seeds = [42, 43, 44, 45, 46]

    # --- Blind/reactive baseline: computed exactly once, independent of lambda and seed ---
    print("Running blind/reactive baseline (unconditional admission, forced latency = 0.0)...")
    baseline_node = QuantumRepeaterNode(T1=T1, T2=T2, depol_prob=depol_prob, shots=shots, seed=quantum_seed)
    baseline_orchestrator = DigitalTwinOrchestrator(model=None, quantum_node=baseline_node,
                                                      threshold=threshold, device=device)
    baseline_metrics = baseline_orchestrator.run_blind_baseline(X_test, y_test)
    print(f"  Baseline: Attempts={baseline_metrics['attempted']} | "
          f"Useful pairs={baseline_metrics['useful_pairs']} | "
          f"Forced latency={baseline_metrics['avg_classical_latency_s']*1000:.4f} ms | "
          f"Confusion (unconditional admission) TP={baseline_metrics['confusion_matrix']['TP']} "
          f"FP={baseline_metrics['confusion_matrix']['FP']} (every dead photon is admitted, by construction)\n")

    rows = []
    per_seed_results = {}

    for lam in lambda_values:
        print(f"[lambda_penalty={lam}] training EdgeLSTM on {len(seeds)} seeds "
              f"({epochs} epochs each) ...")

        seed_runs = []
        for seed in seeds:
            # Seed applied BEFORE model construction: guarantees each round
            # starts from an independent weight initialization.
            torch.manual_seed(seed)
            model = EdgeLSTM(input_size=2, hidden_size=hidden_size, num_layers=1).to(device)
            model = train_edge_lstm(
                model, X_train, y_train,
                threshold=threshold, lambda_penalty=lam, lambda_fn=lambda_fn,
                discard_penalty_weight=discard_penalty_weight, max_discard_rate=max_discard_rate,
                epochs=epochs, lr=lr, device=device, seed=seed, verbose=False,
            )

            # Each round uses its own QuantumRepeaterNode with the SAME
            # quantum_seed: isolates the observed variation to the
            # EdgeLSTM's initialization/training, not to the quantum
            # simulator (which must remain comparable across seeds and
            # across lambdas).
            quantum_node = QuantumRepeaterNode(T1=T1, T2=T2, depol_prob=depol_prob,
                                                shots=shots, seed=quantum_seed)
            orchestrator = DigitalTwinOrchestrator(model=model, quantum_node=quantum_node,
                                                     threshold=threshold, device=device)
            metrics = orchestrator.run_intelligent(X_test, y_test)

            # Pure ML regression quality of F_hat(t) -- computed via one
            # full-batch forward pass over X_test, entirely decoupled from
            # the admission/quantum loop above (see metrics.py).
            regression_metrics = evaluate_predictor_regression(model, X_test, y_test, device=device)

            # Admission confusion matrix (this lambda's FP/FN trade-off --
            # the direct, countable justification for CS_MSELoss's
            # asymmetric penalty) and derived rates.
            confusion = metrics["confusion_matrix"]
            confusion_rates = compute_confusion_metrics(confusion)

            # Dimensionless temporal-scale ratio: replaces a raw-ms
            # latency comparison with a physical constraint against T2.
            latency_ratio_c = compute_latency_ratio(metrics["avg_classical_latency_s"], T2)

            yield_qpu_pct = (metrics["useful_pairs"] / max(metrics["attempted"], 1)) * 100.0
            deficit_surplus = metrics["useful_pairs"] - baseline_metrics["useful_pairs"]

            seed_runs.append({
                "seed": seed,
                "halted": metrics["halted"],
                "attempted": metrics["attempted"],
                "useful_pairs": metrics["useful_pairs"],
                "yield_qpu_pct": yield_qpu_pct,
                "deficit_surplus": deficit_surplus,
                "avg_inference_latency_ms": metrics["avg_classical_latency_s"] * 1000.0,
                "latency_ratio_c": latency_ratio_c,
                "tp": confusion["TP"], "fp": confusion["FP"],
                "tn": confusion["TN"], "fn": confusion["FN"],
                "precision": confusion_rates["precision"], "recall": confusion_rates["recall"],
                "mae": regression_metrics["mae"], "rmse": regression_metrics["rmse"],
                "r2": regression_metrics["r2"],
            })

        per_seed_results[lam] = seed_runs

        halted_mean, halted_std = _mean_std([r["halted"] for r in seed_runs])
        attempted_mean, attempted_std = _mean_std([r["attempted"] for r in seed_runs])
        useful_mean, useful_std = _mean_std([r["useful_pairs"] for r in seed_runs])
        yield_mean, yield_std = _mean_std([r["yield_qpu_pct"] for r in seed_runs])
        deficit_mean, deficit_std = _mean_std([r["deficit_surplus"] for r in seed_runs])
        latency_mean, latency_std = _mean_std([r["avg_inference_latency_ms"] for r in seed_runs])
        latency_ratio_mean, latency_ratio_std = _mean_std([r["latency_ratio_c"] for r in seed_runs])
        fp_mean, fp_std = _mean_std([r["fp"] for r in seed_runs])
        fn_mean, fn_std = _mean_std([r["fn"] for r in seed_runs])
        mae_mean, mae_std = _mean_std([r["mae"] for r in seed_runs])
        rmse_mean, rmse_std = _mean_std([r["rmse"] for r in seed_runs])
        r2_values = [r["r2"] for r in seed_runs if r["r2"] == r["r2"]]  # drop NaN
        r2_mean, r2_std = _mean_std(r2_values) if r2_values else (float("nan"), float("nan"))

        rows.append({
            "Lambda": lam,
            "N Seeds": len(seeds),
            "Cycles Saved (HALT)": f"{halted_mean:.1f} +/- {halted_std:.1f}",
            "QPU Attempts": f"{attempted_mean:.1f} +/- {attempted_std:.1f}",
            "Useful Pairs": f"{useful_mean:.1f} +/- {useful_std:.1f}",
            "QPU Yield (%)": f"{yield_mean:.2f} +/- {yield_std:.2f}",
            "SKR Deficit/Surplus": f"{deficit_mean:+.1f} +/- {deficit_std:.1f}",
            "Inference Latency (ms)": f"{latency_mean:.4f} +/- {latency_std:.4f}",
            "C_latencia (tau_inf/T2)": f"{latency_ratio_mean:.4f} +/- {latency_ratio_std:.4f}",
            "FP (dead photon admitted)": f"{fp_mean:.1f} +/- {fp_std:.1f}",
            "FN (good photon discarded)": f"{fn_mean:.1f} +/- {fn_std:.1f}",
            "MAE": f"{mae_mean:.4f} +/- {mae_std:.4f}",
            "RMSE": f"{rmse_mean:.4f} +/- {rmse_std:.4f}",
            "R2": f"{r2_mean:.4f} +/- {r2_std:.4f}",
        })

        print(f"  -> QPU Yield (mean +/- std) = {yield_mean:.2f}% +/- {yield_std:.2f}% | "
              f"Deficit/Surplus (mean) = {deficit_mean:+.1f} | "
              f"FP (mean) = {fp_mean:.1f} | FN (mean) = {fn_mean:.1f} | "
              f"C_latencia (mean) = {latency_ratio_mean:.4f} | "
              f"MAE (mean) = {mae_mean:.4f} | R2 (mean) = {r2_mean:.4f}\n")

    results_df = pd.DataFrame(rows, columns=[
        "Lambda", "N Seeds", "Cycles Saved (HALT)", "QPU Attempts",
        "Useful Pairs", "QPU Yield (%)", "SKR Deficit/Surplus",
        "Inference Latency (ms)", "C_latencia (tau_inf/T2)",
        "FP (dead photon admitted)", "FN (good photon discarded)",
        "MAE", "RMSE", "R2",
    ])
    return results_df, baseline_metrics, per_seed_results
