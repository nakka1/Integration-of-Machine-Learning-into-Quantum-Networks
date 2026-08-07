"""
Component 7 -- Cross-architecture model comparison
(`run_model_comparison`).

Extends the single-hyperparameter Pareto sweep (`pareto_sweep.py`, which
only varies `lambda_penalty` for a fixed `EdgeLSTM + CS_MSELoss`) to a
comparison ACROSS predictor architectures/objectives:

    - EdgeLSTM + CS_MSELoss  (at one representative `lambda_penalty`,
                                `ComparisonConfig.representative_lambda`)
    - LSTM + MSE              (`baselines.train_lstm_mse`)
    - Random Forest            (`baselines.train_random_forest`)
    - XGBoost                   (`baselines.train_xgboost`, skipped with a
                                  warning if not installed)
    - Transformer                (`baselines.train_transformer`)
    - Blind/reactive baseline     (`orchestrator.run_blind_baseline`,
                                    computed once, same as in
                                    `pareto_sweep.py`)

Every predictor is trained/evaluated over the SAME `seeds` and run
through the SAME `DigitalTwinOrchestrator.run_intelligent` loop against
the SAME `QuantumRepeaterNode` configuration -- so the resulting
throughput / QPU-economy / energy / latency numbers are directly
comparable, and `metrics.build_decision_matrix` can rank them on equal
footing.
"""

from __future__ import annotations

import statistics as stats
import warnings
from typing import List, Sequence

import pandas as pd
import torch

from .baselines import TinyTransformer, train_lstm_mse, train_random_forest, train_transformer
from .config import BaselineConfig, ComparisonConfig, EnergyConfig, QuantumConfig, TrainConfig
from .models import EdgeLSTM, train_edge_lstm
from .metrics import build_decision_matrix, compute_energy_report, compute_qpu_economy, compute_throughput
from .orchestrator import DigitalTwinOrchestrator
from .quantum_node import QuantumRepeaterNode


def _mean_std(values: Sequence[float]) -> tuple:
    mean = stats.fmean(values)
    std = stats.pstdev(values) if len(values) > 1 else 0.0
    return mean, std


def _build_predictor(name: str, seed: int, device: torch.device,
                      X_train: torch.Tensor, y_train: torch.Tensor,
                      train_cfg: TrainConfig, baseline_cfg: BaselineConfig,
                      representative_lambda: float):
    """
    Trains one predictor of `name` for one `seed` and returns an object
    compatible with `DigitalTwinOrchestrator` (a trained `nn.Module`, or a
    `_SklearnRegressorAdapter` for the tree-ensemble baselines).

    Raises `ImportError` for "XGBoost" when `xgboost` isn't installed --
    callers are expected to catch this and skip the model (see
    `run_model_comparison`).
    """
    torch.manual_seed(seed)

    if name == "EdgeLSTM+CS-MSE":
        model = EdgeLSTM(input_size=2, hidden_size=train_cfg.hidden_size, num_layers=1).to(device)
        return train_edge_lstm(
            model, X_train, y_train, threshold=train_cfg.threshold,
            lambda_penalty=representative_lambda, lambda_fn=train_cfg.lambda_fn,
            discard_penalty_weight=train_cfg.discard_penalty_weight,
            max_discard_rate=train_cfg.max_discard_rate,
            epochs=train_cfg.epochs, lr=train_cfg.lr, device=device, seed=seed,
        )

    if name == "LSTM+MSE":
        model = EdgeLSTM(input_size=2, hidden_size=baseline_cfg.lstm_mse_hidden_size, num_layers=1).to(device)
        return train_lstm_mse(model, X_train, y_train, epochs=baseline_cfg.lstm_mse_epochs,
                               lr=baseline_cfg.lstm_mse_lr, device=device, seed=seed)

    if name == "RandomForest":
        return train_random_forest(X_train, y_train, n_estimators=baseline_cfg.rf_n_estimators,
                                    max_depth=baseline_cfg.rf_max_depth, seed=seed)

    if name == "XGBoost":
        from .baselines import train_xgboost  # local import: surfaces ImportError to the caller
        return train_xgboost(X_train, y_train, n_estimators=baseline_cfg.xgb_n_estimators,
                              max_depth=baseline_cfg.xgb_max_depth,
                              learning_rate=baseline_cfg.xgb_learning_rate, seed=seed)

    if name == "Transformer":
        model = TinyTransformer(
            input_size=2, d_model=baseline_cfg.transformer_d_model,
            nhead=baseline_cfg.transformer_nhead, num_layers=baseline_cfg.transformer_num_layers,
            dim_feedforward=baseline_cfg.transformer_dim_feedforward,
        ).to(device)
        return train_transformer(model, X_train, y_train, epochs=baseline_cfg.transformer_epochs,
                                  lr=baseline_cfg.transformer_lr, device=device, seed=seed)

    raise ValueError(f"Unknown predictor name: {name!r}")


def run_model_comparison(X_train: torch.Tensor, y_train: torch.Tensor,
                          X_test: torch.Tensor, y_test: torch.Tensor, device: torch.device,
                          train_cfg: TrainConfig = None, quantum_cfg: QuantumConfig = None,
                          baseline_cfg: BaselineConfig = None, energy_cfg: EnergyConfig = None,
                          comparison_cfg: ComparisonConfig = None,
                          model_names: List[str] = None):
    """
    Trains/evaluates every predictor baseline (plus the blind baseline)
    over `comparison_cfg.seeds`, and reports throughput, QPU economy,
    energy, and a ranked multi-criteria decision matrix.

    Parameters
    ----------
    model_names : list[str], optional
        Subset/order of predictors to include. Default: all five
        (`["EdgeLSTM+CS-MSE", "LSTM+MSE", "RandomForest", "XGBoost",
        "Transformer"]`). "XGBoost" is silently skipped (with a printed
        warning) if `xgboost` isn't installed or
        `comparison_cfg.include_xgboost` is `False`.

    Returns
    -------
    results_df : pd.DataFrame
        One row per model (mean +/- std across seeds): QPU yield, useful
        pairs, throughput, QPU cycles saved, energy saved, inference
        latency.
    baseline_metrics : dict
        Blind/reactive baseline metrics (see `orchestrator.run_blind_baseline`).
    decision_matrix_df : pd.DataFrame
        Output of `metrics.build_decision_matrix` -- normalized criteria,
        weighted `Decision Score`, and `Rank` (1 = recommended model),
        built from the SAME mean values reported in `results_df`.
    per_model_seed_results : dict[str, list[dict]]
        Raw per-seed metrics for each model, preserved for auditing.
    """
    train_cfg = train_cfg or TrainConfig()
    quantum_cfg = quantum_cfg or QuantumConfig()
    baseline_cfg = baseline_cfg or BaselineConfig()
    energy_cfg = energy_cfg or EnergyConfig()
    comparison_cfg = comparison_cfg or ComparisonConfig()
    model_names = model_names or ["EdgeLSTM+CS-MSE", "LSTM+MSE", "RandomForest", "XGBoost", "Transformer"]

    # --- Blind/reactive baseline: computed exactly once ---
    print("Running blind/reactive baseline (unconditional admission, forced latency = 0.0)...")
    baseline_node = QuantumRepeaterNode(T1=quantum_cfg.T1, T2=quantum_cfg.T2, depol_prob=quantum_cfg.depol_prob,
                                         shots=quantum_cfg.shots, seed=quantum_cfg.seed)
    baseline_orchestrator = DigitalTwinOrchestrator(model=None, quantum_node=baseline_node,
                                                      threshold=train_cfg.threshold, device=device)
    baseline_metrics = baseline_orchestrator.run_blind_baseline(X_test, y_test)
    print(f"  Baseline: Attempts={baseline_metrics['attempted']} | "
          f"Useful pairs={baseline_metrics['useful_pairs']}\n")

    rows = []
    per_model_seed_results = {}

    for name in model_names:
        if name == "XGBoost" and not comparison_cfg.include_xgboost:
            print("[XGBoost] skipped (ComparisonConfig.include_xgboost=False).\n")
            continue

        print(f"[{name}] training on {len(comparison_cfg.seeds)} seeds ...")
        seed_runs = []
        skipped = False

        for seed in comparison_cfg.seeds:
            try:
                model = _build_predictor(name, seed, device, X_train, y_train,
                                          train_cfg, baseline_cfg, comparison_cfg.representative_lambda)
            except ImportError as exc:
                warnings.warn(f"[{name}] skipped: {exc}")
                skipped = True
                break

            quantum_node = QuantumRepeaterNode(T1=quantum_cfg.T1, T2=quantum_cfg.T2,
                                                depol_prob=quantum_cfg.depol_prob,
                                                shots=quantum_cfg.shots, seed=quantum_cfg.seed)
            orchestrator = DigitalTwinOrchestrator(model=model, quantum_node=quantum_node,
                                                     threshold=train_cfg.threshold, device=device)
            metrics = orchestrator.run_intelligent(X_test, y_test)

            throughput = compute_throughput(metrics, cycle_time_s=comparison_cfg.cycle_time_s)
            qpu_economy = compute_qpu_economy(metrics, baseline_metrics, shots_per_attempt=quantum_cfg.shots)
            energy = compute_energy_report(metrics, baseline_metrics, shots=quantum_cfg.shots, energy_cfg=energy_cfg)

            yield_qpu_pct = (metrics["useful_pairs"] / max(metrics["attempted"], 1)) * 100.0

            seed_runs.append({
                "seed": seed,
                "useful_pairs": metrics["useful_pairs"],
                "attempted": metrics["attempted"],
                "halted": metrics["halted"],
                "qpu_yield_pct": yield_qpu_pct,
                "deficit_surplus": qpu_economy["useful_pairs_deficit_surplus"],
                "inference_latency_ms": metrics["avg_classical_latency_s"] * 1000.0,
                "throughput_pairs_per_s": throughput["throughput_pairs_per_s"],
                "qpu_cycles_saved": qpu_economy["qpu_cycles_saved"],
                "qpu_cycles_saved_pct": qpu_economy["qpu_cycles_saved_pct"],
                "total_energy_j": energy["total_energy_j"],
                "energy_saved_pct": energy["energy_saved_pct"],
            })

        if skipped or not seed_runs:
            continue

        per_model_seed_results[name] = seed_runs

        def _agg(key):
            return _mean_std([r[key] for r in seed_runs])

        yield_mean, yield_std = _agg("qpu_yield_pct")
        deficit_mean, deficit_std = _agg("deficit_surplus")
        latency_mean, latency_std = _agg("inference_latency_ms")
        throughput_mean, throughput_std = _agg("throughput_pairs_per_s")
        cycles_saved_pct_mean, cycles_saved_pct_std = _agg("qpu_cycles_saved_pct")
        energy_mean, energy_std = _agg("total_energy_j")
        energy_saved_pct_mean, energy_saved_pct_std = _agg("energy_saved_pct")
        useful_mean, useful_std = _agg("useful_pairs")

        rows.append({
            "Model": name,
            "N Seeds": len(seed_runs),
            "Useful Pairs": f"{useful_mean:.1f} +/- {useful_std:.1f}",
            "QPU Yield (%)": f"{yield_mean:.2f} +/- {yield_std:.2f}",
            "SKR Deficit/Surplus": f"{deficit_mean:+.1f} +/- {deficit_std:.1f}",
            "Throughput (pairs/s)": f"{throughput_mean:.2f} +/- {throughput_std:.2f}",
            "QPU Cycles Saved (%)": f"{cycles_saved_pct_mean:.2f} +/- {cycles_saved_pct_std:.2f}",
            "Energy (J)": f"{energy_mean:.6f} +/- {energy_std:.6f}",
            "Energy Saved (%)": f"{energy_saved_pct_mean:+.2f} +/- {energy_saved_pct_std:.2f}",
            "Inference Latency (ms)": f"{latency_mean:.4f} +/- {latency_std:.4f}",
            # Raw means, kept alongside the formatted strings above for
            # `build_decision_matrix` (which needs numeric values).
            "_qpu_yield_pct": yield_mean,
            "_throughput_pairs_per_s": throughput_mean,
            "_qpu_cycles_saved_pct": cycles_saved_pct_mean,
            "_energy_saved_pct": energy_saved_pct_mean,
            "_inference_latency_ms": latency_mean,
        })

        print(f"  -> QPU Yield (mean) = {yield_mean:.2f}% | Throughput (mean) = {throughput_mean:.2f} pairs/s | "
              f"QPU cycles saved (mean) = {cycles_saved_pct_mean:.2f}% | "
              f"Energy saved (mean) = {energy_saved_pct_mean:+.2f}% | "
              f"Latency (mean) = {latency_mean:.4f} ms\n")

    results_df = pd.DataFrame(rows, columns=[
        "Model", "N Seeds", "Useful Pairs", "QPU Yield (%)", "SKR Deficit/Surplus",
        "Throughput (pairs/s)", "QPU Cycles Saved (%)", "Energy (J)", "Energy Saved (%)",
        "Inference Latency (ms)",
    ])

    decision_rows = [{
        "Model": r["Model"],
        "qpu_yield_pct": r["_qpu_yield_pct"],
        "throughput_pairs_per_s": r["_throughput_pairs_per_s"],
        "qpu_cycles_saved_pct": r["_qpu_cycles_saved_pct"],
        "energy_saved_pct": r["_energy_saved_pct"],
        "inference_latency_ms": r["_inference_latency_ms"],
    } for r in rows]
    decision_matrix_df = build_decision_matrix(decision_rows, comparison_cfg.decision_weights, model_key="Model")

    return results_df, baseline_metrics, decision_matrix_df, per_model_seed_results
