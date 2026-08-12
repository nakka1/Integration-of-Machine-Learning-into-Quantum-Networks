"""
Quantum metrics: QPU-time economy relative to the blind/reactive
baseline, and descriptive statistics of the channel's true fidelity
trajectory itself (mean/spread, fraction below threshold, count of
degradation/recovery events).

`useful_pairs` and `QPU Yield (%)` (useful_pairs / attempted) are already
reported directly by `DigitalTwinOrchestrator.run_intelligent` /
`run_pareto_sweep` / `model_comparison.run_model_comparison`; this module
adds the metrics that are DERIVED from those, plus the physical channel
statistics needed to characterize a WDM/Ornstein-Uhlenbeck run
independently of any predictor.
"""

from __future__ import annotations

import numpy as np

from .prediction import find_threshold_crossings


def compute_qpu_economy(metrics: dict, baseline_metrics: dict, shots_per_attempt: int = None) -> dict:
    """
    QPU-time economy relative to the blind/reactive baseline.

    The blind baseline purifies unconditionally on every cycle
    (`baseline_metrics["attempted"] == baseline_metrics["total_steps"]`).
    A predictive controller instead HALTs low-fidelity cycles, so its
    `attempted` count -- and therefore the number of BBPSSW circuits
    actually dispatched to the QPU -- is lower. This function reports both
    the absolute and percentage reduction in QPU attempts (a direct proxy
    for QPU-time budget saved), alongside the resulting useful-pair
    deficit/surplus.

    `shots_per_attempt` (typically `QuantumConfig.shots`) is optional: when
    given, the avoided attempts are also translated into avoided QPU shots
    (`qpu_shots_saved`); when omitted, `qpu_shots_saved` is left as `None`.
    """
    baseline_attempted = max(baseline_metrics["attempted"], 1)
    cycles_saved = baseline_metrics["attempted"] - metrics["attempted"]
    cycles_saved_pct = (cycles_saved / baseline_attempted) * 100.0
    shots_saved = cycles_saved * shots_per_attempt if shots_per_attempt is not None else None
    return {
        "qpu_cycles_saved": cycles_saved,
        "qpu_cycles_saved_pct": cycles_saved_pct,
        "qpu_shots_saved": shots_saved,
        "useful_pairs_deficit_surplus": metrics["useful_pairs"] - baseline_metrics["useful_pairs"],
    }


def compute_fidelity_statistics(y_true, threshold: float = 0.65) -> dict:
    """
    Descriptive statistics of the TRUE fidelity trajectory F(t) itself --
    independent of any predictor -- characterizing how challenging a given
    channel run actually is:

        - mean / std / min / max fidelity over the window.
        - pct_below_threshold : fraction of samples that are already
          "dead" photons by the admission threshold.
        - n_degradation_events / n_recovery_events : count of falling /
          rising threshold crossings (via `find_threshold_crossings`) --
          how often the channel actually flips state, as opposed to
          drifting slowly around the threshold once. A channel with many
          degradation events in a short window is intrinsically harder to
          predict than one with a single, gradual decline.
    """
    y = np.asarray(y_true, dtype=float).ravel()
    if y.size == 0:
        raise ValueError("compute_fidelity_statistics: y_true is empty.")

    degradation_events = find_threshold_crossings(y, threshold, direction="falling")
    recovery_events = find_threshold_crossings(y, threshold, direction="rising")

    return {
        "mean_fidelity": float(np.mean(y)),
        "std_fidelity": float(np.std(y)),
        "min_fidelity": float(np.min(y)),
        "max_fidelity": float(np.max(y)),
        "pct_below_threshold": float(np.mean(y < threshold) * 100.0),
        "n_degradation_events": int(len(degradation_events)),
        "n_recovery_events": int(len(recovery_events)),
    }
