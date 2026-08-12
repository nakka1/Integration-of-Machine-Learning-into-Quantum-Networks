"""
Component 9 -- Ablation study (`run_ablation_study`).

Isolates the individual and combined contribution of the two components
proposed by this work -- the `EdgeLSTM` architecture and the `CS_MSELoss`
cost-sensitive loss function -- via a full 2x2 factorial grid:

    +-------------+------------------+------------------+
    |             | MSE (plain)      | CS-MSE            |
    +-------------+------------------+------------------+
    | StandardLSTM| StandardLSTM+MSE | StandardLSTM+CS-MSE|
    | EdgeLSTM    | EdgeLSTM+MSE     | EdgeLSTM+CS-MSE     |
    +-------------+------------------+------------------+

Every cell is trained/evaluated under the EXACT SAME multi-seed protocol,
`DigitalTwinOrchestrator` loop, and `QuantumRepeaterNode` configuration as
`pareto_sweep.run_pareto_sweep` / `model_comparison.run_model_comparison`,
so results are directly, fairly comparable across cells.

2x2 factorial decomposition
----------------------------
For any numeric metric Y (e.g. `qpu_yield_pct`, `mae`, `fp`), let
`Y(A, L)` denote its mean value (across seeds) at architecture `A` in
{EdgeLSTM, StandardLSTM} and loss `L` in {MSE, CS-MSE}. This module
reports the standard factorial main-effects/interaction decomposition:

    Architecture effect = ((Y(Edge,MSE) + Y(Edge,CS-MSE)) / 2)
                         - ((Y(Std,MSE)  + Y(Std,CS-MSE))  / 2)

        "Holding the loss function fixed (averaged over both), how much
        does swapping StandardLSTM for EdgeLSTM change Y?" Answers
        "What is the impact of the EdgeLSTM architecture?"

    Loss effect         = ((Y(Edge,CS-MSE) + Y(Std,CS-MSE)) / 2)
                         - ((Y(Edge,MSE)    + Y(Std,MSE))    / 2)

        "Holding the architecture fixed (averaged over both), how much
        does swapping MSE for CS-MSE change Y?" Answers "What is the
        impact of the CS-MSE cost function?"

    Interaction effect  = (Y(Edge,CS-MSE) - Y(Edge,MSE))
                         - (Y(Std,CS-MSE)  - Y(Std,MSE))

        "Is the loss function's effect on Y THE SAME regardless of
        architecture, or does EdgeLSTM benefit from CS-MSE by a different
        amount than StandardLSTM does?" A near-zero interaction means the
        two components' contributions are simply additive (the combined
        EdgeLSTM+CS-MSE gain is fully explained by summing the two main
        effects); a large interaction means the components have a
        genuine synergistic (or antagonistic) combined effect that
        neither one alone would predict. Answers "Does the gain come from
        the COMBINATION of the two components?"

This is exactly the classical 2-factor ANOVA effect decomposition, here
applied to point-estimate means (not to per-seed variance/significance --
`decomposition_df` reports the per-seed spread of each cell alongside the
effects so the reader can judge whether the differences are large
relative to seed noise, without this module performing a formal
hypothesis test).
"""

from __future__ import annotations

import statistics as stats
from typing import List, Sequence

import pandas as pd
import torch

from .baselines import train_lstm_mse
from .config import AblationConfig, ComparisonConfig, EnergyConfig, QuantumConfig, TrainConfig
from .models import EdgeLSTM, StandardLSTM, train_edge_lstm
from .metrics import (
    compute_confusion_metrics,
    compute_energy_report,
    compute_latency_ratio,
    compute_qpu_economy,
    compute_throughput,
    evaluate_predictor_regression,
)
from .orchestrator import DigitalTwinOrchestrator
from .quantum_node import QuantumRepeaterNode

ARCHITECTURES = ["EdgeLSTM", "StandardLSTM"]
LOSSES = ["MSE", "CS-MSE"]

# Metrics where a LOWER value is better -- used only to phrase the
# human-readable interpretation strings; the raw effect numbers in
# `decomposition_df` are unaffected by this and always follow the sign
# convention documented in the module docstring (Edge/CS-MSE side minus
# StandardLSTM/MSE side).
_LOWER_IS_BETTER = {"mae", "rmse", "fp", "fn"}


def _mean_std(values: Sequence[float]) -> tuple:
    mean = stats.fmean(values)
    std = stats.pstdev(values) if len(values) > 1 else 0.0
    return mean, std


def _mean_std_dropna(values: Sequence[float]) -> tuple:
    clean = [v for v in values if v == v]  # NaN != NaN
    return _mean_std(clean) if clean else (float("nan"), float("nan"))


def _build_and_train(architecture: str, loss: str, seed: int, device: torch.device,
                      X_train: torch.Tensor, y_train: torch.Tensor,
                      train_cfg: TrainConfig, ablation_cfg: AblationConfig):
    """Builds one (architecture, loss) grid cell's model and trains it for one seed."""
    torch.manual_seed(seed)

    if architecture == "EdgeLSTM":
        model = EdgeLSTM(input_size=2, hidden_size=train_cfg.hidden_size, num_layers=1).to(device)
    elif architecture == "StandardLSTM":
        model = StandardLSTM(
            input_size=2, hidden_size=ablation_cfg.standard_lstm_hidden_size,
            num_layers=ablation_cfg.standard_lstm_num_layers,
            dropout=ablation_cfg.standard_lstm_dropout,
        ).to(device)
    else:
        raise ValueError(f"Unknown architecture: {architecture!r}")

    if loss == "MSE":
        return train_lstm_mse(model, X_train, y_train, epochs=ablation_cfg.epochs,
                               lr=ablation_cfg.lr, device=device, seed=seed)
    if loss == "CS-MSE":
        return train_edge_lstm(
            model, X_train, y_train, threshold=train_cfg.threshold,
            lambda_penalty=ablation_cfg.representative_lambda, lambda_fn=train_cfg.lambda_fn,
            discard_penalty_weight=train_cfg.discard_penalty_weight,
            max_discard_rate=train_cfg.max_discard_rate,
            epochs=ablation_cfg.epochs, lr=ablation_cfg.lr, device=device, seed=seed,
        )
    raise ValueError(f"Unknown loss: {loss!r}")


def run_ablation_study(X_train: torch.Tensor, y_train: torch.Tensor,
                        X_test: torch.Tensor, y_test: torch.Tensor, device: torch.device,
                        train_cfg: TrainConfig = None, quantum_cfg: QuantumConfig = None,
                        ablation_cfg: AblationConfig = None, energy_cfg: EnergyConfig = None,
                        comparison_cfg: ComparisonConfig = None):
    """
    Trains/evaluates all four cells of the `{StandardLSTM, EdgeLSTM} x
    {MSE, CS-MSE}` grid over `ablation_cfg.seeds`, reporting the same rich
    metrics suite as `model_comparison.run_model_comparison` per cell
    (mean +/- std across seeds), plus the 2x2 factorial decomposition for
    every metric in `ablation_cfg.headline_metrics`.

    Returns
    -------
    results_df : pd.DataFrame
        One row per grid cell (`Architecture`, `Loss`), same metric
        columns as `run_model_comparison`'s `results_df`.
    decomposition_df : pd.DataFrame
        One row per headline metric: `Architecture Effect`, `Loss Effect`,
        `Interaction Effect` (raw, signed -- see module docstring), plus a
        human-readable `Interpretation` string.
    baseline_metrics : dict
        Blind/reactive baseline metrics, computed once.
    per_cell_seed_results : dict[(str, str), list[dict]]
        Raw per-seed metrics for each (architecture, loss) cell, keyed by
        e.g. `("EdgeLSTM", "CS-MSE")`, preserved for auditing.
    """
    train_cfg = train_cfg or TrainConfig()
    quantum_cfg = quantum_cfg or QuantumConfig()
    ablation_cfg = ablation_cfg or AblationConfig()
    energy_cfg = energy_cfg or EnergyConfig()
    comparison_cfg = comparison_cfg or ComparisonConfig()

    print("Running blind/reactive baseline (unconditional admission, forced latency = 0.0)...")
    baseline_node = QuantumRepeaterNode(T1=quantum_cfg.T1, T2=quantum_cfg.T2, depol_prob=quantum_cfg.depol_prob,
                                         shots=quantum_cfg.shots, seed=quantum_cfg.seed)
    baseline_orchestrator = DigitalTwinOrchestrator(model=None, quantum_node=baseline_node,
                                                      threshold=train_cfg.threshold, device=device)
    baseline_metrics = baseline_orchestrator.run_blind_baseline(X_test, y_test)
    print(f"  Baseline: Attempts={baseline_metrics['attempted']} | "
          f"Useful pairs={baseline_metrics['useful_pairs']}\n")

    rows = []
    per_cell_seed_results = {}
    cell_metric_means = {}  # (architecture, loss) -> {metric_name: mean_value}

    for architecture in ARCHITECTURES:
        for loss in LOSSES:
            cell_name = f"{architecture}+{loss}"
            print(f"[{cell_name}] training on {len(ablation_cfg.seeds)} seeds "
                  f"({ablation_cfg.epochs} epochs each) ...")

            seed_runs = []
            for seed in ablation_cfg.seeds:
                model = _build_and_train(architecture, loss, seed, device, X_train, y_train,
                                          train_cfg, ablation_cfg)

                regression_metrics = evaluate_predictor_regression(model, X_test, y_test, device=device)

                quantum_node = QuantumRepeaterNode(T1=quantum_cfg.T1, T2=quantum_cfg.T2,
                                                    depol_prob=quantum_cfg.depol_prob,
                                                    shots=quantum_cfg.shots, seed=quantum_cfg.seed)
                orchestrator = DigitalTwinOrchestrator(model=model, quantum_node=quantum_node,
                                                         threshold=train_cfg.threshold, device=device)
                metrics = orchestrator.run_intelligent(X_test, y_test)

                confusion = metrics["confusion_matrix"]
                confusion_rates = compute_confusion_metrics(confusion)
                throughput = compute_throughput(metrics, cycle_time_s=ablation_cfg.cycle_time_s)
                qpu_economy = compute_qpu_economy(metrics, baseline_metrics, shots_per_attempt=quantum_cfg.shots)
                energy = compute_energy_report(metrics, baseline_metrics, shots=quantum_cfg.shots, energy_cfg=energy_cfg)
                latency_ratio_c = compute_latency_ratio(metrics["avg_classical_latency_s"], quantum_cfg.T2)

                yield_qpu_pct = (metrics["useful_pairs"] / max(metrics["attempted"], 1)) * 100.0

                seed_runs.append({
                    "seed": seed,
                    "useful_pairs": metrics["useful_pairs"],
                    "attempted": metrics["attempted"],
                    "halted": metrics["halted"],
                    "qpu_yield_pct": yield_qpu_pct,
                    "deficit_surplus": qpu_economy["useful_pairs_deficit_surplus"],
                    "mae": regression_metrics["mae"],
                    "rmse": regression_metrics["rmse"],
                    "r2": regression_metrics["r2"],
                    "tp": confusion["TP"], "fp": confusion["FP"],
                    "tn": confusion["TN"], "fn": confusion["FN"],
                    "precision": confusion_rates["precision"], "recall": confusion_rates["recall"],
                    "f1": confusion_rates["f1"],
                    "inference_latency_ms": metrics["avg_classical_latency_s"] * 1000.0,
                    "latency_ratio_c": latency_ratio_c,
                    "throughput_pairs_per_s": throughput["throughput_pairs_per_s"],
                    "qpu_cycles_saved_pct": qpu_economy["qpu_cycles_saved_pct"],
                    "total_energy_j": energy["total_energy_j"],
                    "energy_saved_pct": energy["energy_saved_pct"],
                })

            per_cell_seed_results[(architecture, loss)] = seed_runs

            def _agg(key):
                return _mean_std([r[key] for r in seed_runs])

            yield_mean, yield_std = _agg("qpu_yield_pct")
            deficit_mean, deficit_std = _agg("deficit_surplus")
            mae_mean, mae_std = _agg("mae")
            rmse_mean, rmse_std = _agg("rmse")
            r2_mean, r2_std = _mean_std_dropna([r["r2"] for r in seed_runs])
            fp_mean, fp_std = _agg("fp")
            fn_mean, fn_std = _agg("fn")
            precision_mean, precision_std = _mean_std_dropna([r["precision"] for r in seed_runs])
            recall_mean, recall_std = _mean_std_dropna([r["recall"] for r in seed_runs])
            f1_mean, f1_std = _mean_std_dropna([r["f1"] for r in seed_runs])
            latency_mean, latency_std = _agg("inference_latency_ms")
            latency_ratio_mean, latency_ratio_std = _agg("latency_ratio_c")
            throughput_mean, throughput_std = _agg("throughput_pairs_per_s")
            cycles_saved_pct_mean, cycles_saved_pct_std = _agg("qpu_cycles_saved_pct")
            energy_mean, energy_std = _agg("total_energy_j")
            energy_saved_pct_mean, energy_saved_pct_std = _agg("energy_saved_pct")
            useful_mean, useful_std = _agg("useful_pairs")

            cell_metric_means[(architecture, loss)] = {
                "qpu_yield_pct": yield_mean, "mae": mae_mean, "rmse": rmse_mean, "r2": r2_mean,
                "fp": fp_mean, "fn": fn_mean, "precision": precision_mean, "recall": recall_mean,
                "f1": f1_mean, "throughput_pairs_per_s": throughput_mean,
                "qpu_cycles_saved_pct": cycles_saved_pct_mean, "energy_saved_pct": energy_saved_pct_mean,
                "latency_ratio_c": latency_ratio_mean, "useful_pairs": useful_mean,
            }

            rows.append({
                "Architecture": architecture,
                "Loss": loss,
                "Model": cell_name,
                "N Seeds": len(seed_runs),
                "Useful Pairs": f"{useful_mean:.1f} +/- {useful_std:.1f}",
                "QPU Yield (%)": f"{yield_mean:.2f} +/- {yield_std:.2f}",
                "SKR Deficit/Surplus": f"{deficit_mean:+.1f} +/- {deficit_std:.1f}",
                "MAE": f"{mae_mean:.4f} +/- {mae_std:.4f}",
                "RMSE": f"{rmse_mean:.4f} +/- {rmse_std:.4f}",
                "R2": f"{r2_mean:.4f} +/- {r2_std:.4f}",
                "FP (dead photon admitted)": f"{fp_mean:.1f} +/- {fp_std:.1f}",
                "FN (good photon discarded)": f"{fn_mean:.1f} +/- {fn_std:.1f}",
                "Precision": f"{precision_mean:.4f} +/- {precision_std:.4f}",
                "Recall": f"{recall_mean:.4f} +/- {recall_std:.4f}",
                "F1": f"{f1_mean:.4f} +/- {f1_std:.4f}",
                "Throughput (pairs/s)": f"{throughput_mean:.2f} +/- {throughput_std:.2f}",
                "QPU Cycles Saved (%)": f"{cycles_saved_pct_mean:.2f} +/- {cycles_saved_pct_std:.2f}",
                "Energy (J)": f"{energy_mean:.6f} +/- {energy_std:.6f}",
                "Energy Saved (%)": f"{energy_saved_pct_mean:+.2f} +/- {energy_saved_pct_std:.2f}",
                "Inference Latency (ms)": f"{latency_mean:.4f} +/- {latency_std:.4f}",
                "C_latencia (tau_inf/T2)": f"{latency_ratio_mean:.4f} +/- {latency_ratio_std:.4f}",
            })

            print(f"  -> QPU Yield (mean) = {yield_mean:.2f}% | MAE (mean) = {mae_mean:.4f} | "
                  f"FP (mean) = {fp_mean:.1f} | FN (mean) = {fn_mean:.1f}\n")

    results_df = pd.DataFrame(rows, columns=[
        "Architecture", "Loss", "Model", "N Seeds", "Useful Pairs", "QPU Yield (%)",
        "SKR Deficit/Surplus", "MAE", "RMSE", "R2", "FP (dead photon admitted)",
        "FN (good photon discarded)", "Precision", "Recall", "F1", "Throughput (pairs/s)",
        "QPU Cycles Saved (%)", "Energy (J)", "Energy Saved (%)", "Inference Latency (ms)",
        "C_latencia (tau_inf/T2)",
    ])

    decomposition_df = _build_decomposition_df(cell_metric_means, ablation_cfg.headline_metrics)

    return results_df, decomposition_df, baseline_metrics, per_cell_seed_results


def _build_decomposition_df(cell_metric_means: dict, headline_metrics: List[str]) -> pd.DataFrame:
    """
    Builds the 2x2 factorial decomposition table (Architecture Effect /
    Loss Effect / Interaction Effect + a human-readable interpretation)
    for every metric in `headline_metrics`, from the four cells' mean
    values -- see the module docstring for the exact formulas.
    """
    rows = []
    for metric in headline_metrics:
        y_edge_mse = cell_metric_means[("EdgeLSTM", "MSE")][metric]
        y_edge_cs = cell_metric_means[("EdgeLSTM", "CS-MSE")][metric]
        y_std_mse = cell_metric_means[("StandardLSTM", "MSE")][metric]
        y_std_cs = cell_metric_means[("StandardLSTM", "CS-MSE")][metric]

        architecture_effect = ((y_edge_mse + y_edge_cs) / 2.0) - ((y_std_mse + y_std_cs) / 2.0)
        loss_effect = ((y_edge_cs + y_std_cs) / 2.0) - ((y_edge_mse + y_std_mse) / 2.0)
        interaction_effect = (y_edge_cs - y_edge_mse) - (y_std_cs - y_std_mse)

        lower_is_better = metric in _LOWER_IS_BETTER

        def _verdict(effect: float) -> str:
            if effect == 0:
                return "has no effect on"
            improves = (effect < 0) == lower_is_better
            return "improves" if improves else "worsens"

        arch_verdict = _verdict(architecture_effect)
        loss_verdict = _verdict(loss_effect)

        interpretation = (
            f"EdgeLSTM (vs StandardLSTM) {arch_verdict} '{metric}' (architecture effect = "
            f"{architecture_effect:+.4g}). CS-MSE (vs plain MSE) {loss_verdict} '{metric}' "
            f"(loss effect = {loss_effect:+.4g}). Interaction = {interaction_effect:+.4g}: "
            + ("effects are approximately additive (no strong synergy/antagonism detected)."
               if abs(interaction_effect) < 0.1 * max(abs(architecture_effect), abs(loss_effect), 1e-9)
               else "effects are NOT purely additive -- CS-MSE's benefit depends on which "
                    "architecture it is paired with, i.e. part of the gain comes from the "
                    "COMBINATION of the two components, not from either alone.")
        )

        rows.append({
            "Metric": metric,
            "EdgeLSTM+MSE": y_edge_mse,
            "EdgeLSTM+CS-MSE": y_edge_cs,
            "StandardLSTM+MSE": y_std_mse,
            "StandardLSTM+CS-MSE": y_std_cs,
            "Architecture Effect": architecture_effect,
            "Loss Effect": loss_effect,
            "Interaction Effect": interaction_effect,
            "Interpretation": interpretation,
        })

    return pd.DataFrame(rows, columns=[
        "Metric", "EdgeLSTM+MSE", "EdgeLSTM+CS-MSE", "StandardLSTM+MSE", "StandardLSTM+CS-MSE",
        "Architecture Effect", "Loss Effect", "Interaction Effect", "Interpretation",
    ])
