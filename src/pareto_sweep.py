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
"""

from __future__ import annotations

import statistics as stats
from typing import List, Sequence

import pandas as pd
import torch

from .models import EdgeLSTM, train_edge_lstm
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
          f"Forced latency={baseline_metrics['avg_classical_latency_s']*1000:.4f} ms\n")

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
            })

        per_seed_results[lam] = seed_runs

        halted_mean, halted_std = _mean_std([r["halted"] for r in seed_runs])
        attempted_mean, attempted_std = _mean_std([r["attempted"] for r in seed_runs])
        useful_mean, useful_std = _mean_std([r["useful_pairs"] for r in seed_runs])
        yield_mean, yield_std = _mean_std([r["yield_qpu_pct"] for r in seed_runs])
        deficit_mean, deficit_std = _mean_std([r["deficit_surplus"] for r in seed_runs])
        latency_mean, latency_std = _mean_std([r["avg_inference_latency_ms"] for r in seed_runs])

        rows.append({
            "Lambda": lam,
            "N Seeds": len(seeds),
            "Cycles Saved (HALT)": f"{halted_mean:.1f} +/- {halted_std:.1f}",
            "QPU Attempts": f"{attempted_mean:.1f} +/- {attempted_std:.1f}",
            "Useful Pairs": f"{useful_mean:.1f} +/- {useful_std:.1f}",
            "QPU Yield (%)": f"{yield_mean:.2f} +/- {yield_std:.2f}",
            "SKR Deficit/Surplus": f"{deficit_mean:+.1f} +/- {deficit_std:.1f}",
            "Inference Latency (ms)": f"{latency_mean:.4f} +/- {latency_std:.4f}",
        })

        print(f"  -> QPU Yield (mean +/- std) = {yield_mean:.2f}% +/- {yield_std:.2f}% | "
              f"Deficit/Surplus (mean) = {deficit_mean:+.1f} | "
              f"Inference latency (mean) = {latency_mean:.4f} ms\n")

    results_df = pd.DataFrame(rows, columns=[
        "Lambda", "N Seeds", "Cycles Saved (HALT)", "QPU Attempts",
        "Useful Pairs", "QPU Yield (%)", "SKR Deficit/Surplus",
        "Inference Latency (ms)",
    ])
    return results_df, baseline_metrics, per_seed_results
