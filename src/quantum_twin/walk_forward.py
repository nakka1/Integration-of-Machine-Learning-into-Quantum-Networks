"""
Component 14 -- Walk-forward (rolling-origin) temporal cross-validation.

Everywhere else in this project, `WDMChannelSimulator.preprocess` performs
a SINGLE chronological train/test split (the first `1 - test_size`
fraction of windows for training, the rest for testing). That is standard
practice for a quick sweep, but it means every reported metric rests on
exactly one arbitrary boundary in time -- if that particular boundary
happens to land right before/after an unusually easy or hard stretch of
the synthetic channel, the whole result inherits that luck.

`run_walk_forward_evaluation` replaces the single split with several
NON-OVERLAPPING folds that walk forward through the SAME full windowed
dataset (see `generate_walk_forward_splits`), training and evaluating a
fresh `EdgeLSTM + CS_MSELoss` model per fold with the exact same
regression/confusion metrics used throughout the rest of the project, and
aggregates the per-fold results into a 95% confidence interval per metric
(via `statistics_tests.compute_confidence_interval`) -- directly answering
"do the reported gains hold up across different slices of time, or are
they an artifact of the one train/test boundary everywhere else in this
project happens to use?".
"""

from __future__ import annotations

from typing import List, Tuple

import pandas as pd
import torch

from .config import QuantumConfig, TrainConfig, WalkForwardConfig
from .metrics import compute_confusion_metrics, evaluate_predictor_regression
from .models import EdgeLSTM, train_edge_lstm
from .orchestrator import DigitalTwinOrchestrator
from .quantum_node import QuantumRepeaterNode
from .reproducibility import seed_everything
from .statistics_tests import compute_confidence_interval


def generate_walk_forward_splits(n_samples: int, wf_cfg: WalkForwardConfig) -> List[Tuple[range, range]]:
    """
    Returns a list of `(train_range, test_range)` index pairs, one per
    fold, walking forward through `n_samples` in chronological order (see
    `WalkForwardConfig`'s docstring for the exact semantics of
    `expanding`/`gap`).

    Fold `k`'s test segment ends at `min_train_size + (k+1) * test_size +
    gap`; folds are generated in order until either `n_splits` folds have
    been produced or a fold's test segment would run past `n_samples`,
    whichever comes first (so asking for more folds than the data
    supports degrades gracefully to however many DO fit, rather than
    raising -- unless NONE fit at all, which IS an error: see below).

    Raises `ValueError` if not even a single fold fits within `n_samples`
    given `wf_cfg.min_train_size`/`test_size`/`gap` -- silently returning
    zero folds would let a misconfigured call proceed and only fail
    later, confusingly, deep inside `run_walk_forward_evaluation`.
    """
    splits = []
    for k in range(wf_cfg.n_splits):
        train_end = wf_cfg.min_train_size + k * wf_cfg.test_size
        test_start = train_end + wf_cfg.gap
        test_end = test_start + wf_cfg.test_size
        if test_end > n_samples:
            break
        train_start = 0 if wf_cfg.expanding else max(0, train_end - wf_cfg.min_train_size)
        splits.append((range(train_start, train_end), range(test_start, test_end)))

    if not splits:
        raise ValueError(
            f"generate_walk_forward_splits: no fold fits within n_samples={n_samples} given "
            f"min_train_size={wf_cfg.min_train_size}, test_size={wf_cfg.test_size}, gap={wf_cfg.gap} "
            f"(need at least min_train_size + test_size + gap = "
            f"{wf_cfg.min_train_size + wf_cfg.test_size + wf_cfg.gap} samples). "
            f"Reduce these values, increase n_steps in SimConfig, or reduce test_size in preprocess()."
        )
    return splits


def run_walk_forward_evaluation(X_full: torch.Tensor, y_full: torch.Tensor, device: torch.device,
                                 train_cfg: TrainConfig | None = None, quantum_cfg: QuantumConfig | None = None,
                                 wf_cfg: WalkForwardConfig | None = None,
                                 ) -> Tuple[pd.DataFrame, pd.DataFrame, List[Tuple[range, range]]]:
    """
    Trains and evaluates one `EdgeLSTM + CS_MSELoss` model per
    walk-forward fold (see `generate_walk_forward_splits`), reporting the
    same regression (MAE/RMSE/R^2) and admission-confusion
    (precision/recall/F1/FP/FN) metrics used throughout the rest of this
    project, plus QPU yield from the full `DigitalTwinOrchestrator`
    simulation loop.

    Parameters
    ----------
    X_full, y_full : torch.Tensor
        The FULL windowed dataset in chronological order -- i.e. from
        `WDMChannelSimulator.preprocess(df, window_size=..., test_size=0.0)`'s
        `X_train`/`y_train` (with `test_size=0.0`, ALL windows land in the
        "train" output; `run_walk_forward_evaluation` performs its own
        splitting internally via `generate_walk_forward_splits`).

    Returns
    -------
    fold_df : pd.DataFrame
        One row per fold: fold index, train/test segment boundaries, and
        every metric.
    summary_df : pd.DataFrame
        One row per metric: mean and 95% confidence interval ACROSS
        folds (via `statistics_tests.compute_confidence_interval`) -- the
        headline "does this hold up across time?" table.
    splits : list[(range, range)]
        The raw fold boundaries, as returned by
        `generate_walk_forward_splits` -- reused by
        `plotting.plot_walk_forward_folds` to draw the fold timeline.
    """
    train_cfg = train_cfg or TrainConfig()
    quantum_cfg = quantum_cfg or QuantumConfig()
    wf_cfg = wf_cfg or WalkForwardConfig()

    n_samples = X_full.shape[0]
    splits = generate_walk_forward_splits(n_samples, wf_cfg)

    fold_rows = []
    for fold_idx, (train_range, test_range) in enumerate(splits):
        X_train_fold = X_full[train_range.start:train_range.stop].to(device)
        y_train_fold = y_full[train_range.start:train_range.stop].to(device)
        X_test_fold = X_full[test_range.start:test_range.stop].to(device)
        y_test_fold = y_full[test_range.start:test_range.stop].to(device)

        print(f"[Fold {fold_idx}] train=[{train_range.start}:{train_range.stop}) "
              f"({len(train_range)} samples) | test=[{test_range.start}:{test_range.stop}) "
              f"({len(test_range)} samples) ...")

        # Seed offset by fold index: each fold gets a distinct, but fully
        # reproducible, weight initialization (see WalkForwardConfig.seed).
        seed_everything(wf_cfg.seed + fold_idx)
        model = EdgeLSTM(input_size=2, hidden_size=train_cfg.hidden_size, num_layers=1).to(device)
        model = train_edge_lstm(
            model, X_train_fold, y_train_fold, threshold=train_cfg.threshold,
            lambda_penalty=wf_cfg.representative_lambda, lambda_fn=train_cfg.lambda_fn,
            discard_penalty_weight=train_cfg.discard_penalty_weight,
            max_discard_rate=train_cfg.max_discard_rate,
            epochs=wf_cfg.epochs, lr=wf_cfg.lr, device=device, seed=wf_cfg.seed + fold_idx,
        )

        regression_metrics = evaluate_predictor_regression(model, X_test_fold, y_test_fold, device=device)

        quantum_node = QuantumRepeaterNode(T1=quantum_cfg.T1, T2=quantum_cfg.T2, depol_prob=quantum_cfg.depol_prob,
                                            shots=quantum_cfg.shots, seed=quantum_cfg.seed)
        orchestrator = DigitalTwinOrchestrator(model=model, quantum_node=quantum_node,
                                                 threshold=train_cfg.threshold, device=device)
        metrics = orchestrator.run_intelligent(X_test_fold, y_test_fold)
        confusion = metrics["confusion_matrix"]
        confusion_rates = compute_confusion_metrics(confusion)
        yield_pct = (metrics["useful_pairs"] / max(metrics["attempted"], 1)) * 100.0

        fold_rows.append({
            "fold": fold_idx,
            "train_start": train_range.start, "train_end": train_range.stop, "train_size": len(train_range),
            "test_start": test_range.start, "test_end": test_range.stop, "test_size": len(test_range),
            "mae": regression_metrics["mae"], "rmse": regression_metrics["rmse"], "r2": regression_metrics["r2"],
            "qpu_yield_pct": yield_pct, "useful_pairs": metrics["useful_pairs"], "attempted": metrics["attempted"],
            "fp": confusion["FP"], "fn": confusion["FN"], "tp": confusion["TP"], "tn": confusion["TN"],
            "precision": confusion_rates["precision"], "recall": confusion_rates["recall"],
            "f1": confusion_rates["f1"],
        })

        print(f"  -> MAE={regression_metrics['mae']:.4f} | QPU Yield={yield_pct:.2f}% | "
              f"FP={confusion['FP']} | FN={confusion['FN']}\n")

    fold_df = pd.DataFrame(fold_rows, columns=[
        "fold", "train_start", "train_end", "train_size", "test_start", "test_end", "test_size",
        "mae", "rmse", "r2", "qpu_yield_pct", "useful_pairs", "attempted", "fp", "fn", "tp", "tn",
        "precision", "recall", "f1",
    ])

    summary_rows = []
    for metric in ["mae", "rmse", "r2", "qpu_yield_pct", "fp", "fn", "precision", "recall", "f1"]:
        values = [v for v in fold_df[metric] if v == v]  # drop NaN (e.g. degenerate R^2)
        if not values:
            continue
        mean, ci_low, ci_high = compute_confidence_interval(values, confidence=0.95)
        summary_rows.append({"Metric": metric, "Mean": mean, "95% CI Low": ci_low,
                              "95% CI High": ci_high, "N Folds": len(values)})
    summary_df = pd.DataFrame(summary_rows, columns=["Metric", "Mean", "95% CI Low", "95% CI High", "N Folds"])

    return fold_df, summary_df, splits
