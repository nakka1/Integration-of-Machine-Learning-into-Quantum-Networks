"""
Component 6 -- Derived operational metrics and the multi-criteria
decision matrix.

`pareto_sweep.py` already reports QPU yield, useful-pair deficit/surplus,
and inference latency. This module adds the three metrics needed to
actually *decide* which predictor to deploy, plus the decision matrix
itself:

    - `compute_throughput`     : useful entangled pairs delivered per
                                   second of wall-clock operation.
    - `compute_qpu_economy`    : purification attempts (and therefore QPU
                                   time/shots) avoided relative to the
                                   blind/reactive baseline.
    - `compute_energy_report`  : an illustrative Joules accounting of the
                                   quantum-side (per purification attempt)
                                   and classical-side (per prediction)
                                   energy cost, and the resulting savings
                                   relative to the blind baseline.
    - `build_decision_matrix`  : normalizes every criterion (direction-
                                   aware) across candidate models and
                                   computes a weighted composite score.

All three "compute_*" functions accept the same `metrics` dict produced
by `DigitalTwinOrchestrator.run_intelligent` / `run_blind_baseline`
(`total_steps`, `useful_pairs`, `halted`, `attempted`,
`avg_classical_latency_s`), so they compose directly with the existing
pipeline without touching `orchestrator.py`.
"""

from __future__ import annotations

from typing import Dict, List, Sequence

import pandas as pd

from .config import EnergyConfig


# ---------------------------------------------------------------------------
# Throughput
# ---------------------------------------------------------------------------

def compute_throughput(metrics: dict, cycle_time_s: float = 1e-3) -> dict:
    """
    Useful-pair throughput, in pairs/second.

    `cycle_time_s` is the fixed repetition period of one WDM
    entanglement-generation attempt (the channel keeps ticking at this
    rate regardless of the admission decision -- HALT just skips the
    purification step for that cycle, it doesn't stop the clock). Total
    wall-clock time is therefore:

        total_time_s = total_steps * cycle_time_s + total_classical_latency_s

    where `total_classical_latency_s = avg_classical_latency_s *
    total_steps` adds back the classical inference overhead paid on every
    cycle (zero for the blind baseline, which never invokes a predictor).

    Returns a dict with `total_time_s` and `throughput_pairs_per_s`.
    """
    total_steps = max(metrics["total_steps"], 1)
    total_classical_latency_s = metrics["avg_classical_latency_s"] * total_steps
    total_time_s = total_steps * cycle_time_s + total_classical_latency_s
    throughput = metrics["useful_pairs"] / total_time_s if total_time_s > 0 else 0.0
    return {
        "total_time_s": total_time_s,
        "throughput_pairs_per_s": throughput,
    }


# ---------------------------------------------------------------------------
# QPU economy
# ---------------------------------------------------------------------------

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
    deficit/surplus already computed in `pareto_sweep.py` for context.

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


# ---------------------------------------------------------------------------
# Energy accounting
# ---------------------------------------------------------------------------

def _energy_per_attempt_j(energy_cfg: EnergyConfig, shots: int) -> float:
    """
    Illustrative quantum-side energy of ONE BBPSSW purification attempt
    (see `EnergyConfig` docstring for the caveat that these are
    order-of-magnitude coefficients, not datasheet values):

        E_attempt = shots * (n_1q * E_1q + n_2q * E_2q + E_shot_overhead)
    """
    per_shot = (
        energy_cfg.gates_1q_per_attempt * energy_cfg.joules_per_1q_gate
        + energy_cfg.gates_2q_per_attempt * energy_cfg.joules_per_2q_gate
        + energy_cfg.joules_per_shot_overhead
    )
    return shots * per_shot


def compute_energy_report(metrics: dict, baseline_metrics: dict, shots: int,
                           energy_cfg: EnergyConfig = None) -> dict:
    """
    Illustrative total-energy comparison (Joules) between a predictive
    controller and the blind baseline, over the same evaluation window.

    Two components, summed per cycle:

        - Quantum energy : paid ONLY on cycles where a purification
          attempt is actually dispatched (`attempted`), at
          `_energy_per_attempt_j(...)` each.
        - Classical energy : paid on EVERY cycle. For a predictive
          controller, `classical_inference_power_w * avg_classical_latency_s`
          per cycle (the edge accelerator actively running the predictor);
          for the blind baseline (no predictor invoked),
          `classical_idle_power_w * cycle-equivalent latency` -- using the
          predictor's own average latency as the reference cycle length so
          the two totals are computed over comparable wall-clock time.

    Returns quantum/classical/total energy (J) for `metrics`, the
    equivalent total for the baseline, and the resulting percentage
    saved (positive = the predictor used less total energy than blind
    purification).
    """
    energy_cfg = energy_cfg or EnergyConfig()
    e_attempt = _energy_per_attempt_j(energy_cfg, shots)

    quantum_j = metrics["attempted"] * e_attempt
    classical_j = (
        metrics["total_steps"] * metrics["avg_classical_latency_s"]
        * energy_cfg.classical_inference_power_w
    )
    total_j = quantum_j + classical_j

    baseline_quantum_j = baseline_metrics["attempted"] * e_attempt
    # Reference idle window: the predictor's own average latency, so the
    # baseline's classical term is on the same time basis as `metrics`
    # rather than implicitly zero.
    baseline_classical_j = (
        baseline_metrics["total_steps"] * metrics["avg_classical_latency_s"]
        * energy_cfg.classical_idle_power_w
    )
    baseline_total_j = baseline_quantum_j + baseline_classical_j

    energy_saved_j = baseline_total_j - total_j
    energy_saved_pct = (energy_saved_j / baseline_total_j * 100.0) if baseline_total_j > 0 else 0.0

    return {
        "quantum_energy_j": quantum_j,
        "classical_energy_j": classical_j,
        "total_energy_j": total_j,
        "baseline_total_energy_j": baseline_total_j,
        "energy_saved_j": energy_saved_j,
        "energy_saved_pct": energy_saved_pct,
    }


# ---------------------------------------------------------------------------
# Decision matrix
# ---------------------------------------------------------------------------

# Criteria where a LOWER raw value is better (inverted before weighting).
_COST_CRITERIA = {"inference_latency_ms", "total_energy_j"}


def build_decision_matrix(rows: List[dict], weights: Dict[str, float],
                           model_key: str = "Model") -> pd.DataFrame:
    """
    Multi-criteria decision matrix: normalizes every weighted criterion
    to [0, 1] (min-max, direction-aware -- see `_COST_CRITERIA`) across
    the candidate models in `rows`, then computes a weighted composite
    score in [0, 1] (higher is better).

    Parameters
    ----------
    rows : list[dict]
        One dict per candidate model, each containing `model_key` plus
        every key present in `weights` (raw, un-normalized values -- e.g.
        `qpu_yield_pct`, `throughput_pairs_per_s`, `qpu_cycles_saved_pct`,
        `energy_saved_pct`, `inference_latency_ms`).
    weights : dict[str, float]
        Criterion -> weight (non-negative; renormalized to sum to 1
        internally, so callers don't need to pre-normalize).
    model_key : str
        Column name identifying each row's model (default "Model").

    Returns
    -------
    pd.DataFrame
        One row per model, with a `<criterion>_norm` column per weighted
        criterion, a `Decision Score` column (weighted sum, higher =
        better), and a `Rank` column (1 = best), sorted by rank.
    """
    if not rows:
        return pd.DataFrame(columns=[model_key, "Decision Score", "Rank"])

    total_weight = sum(max(w, 0.0) for w in weights.values())
    if total_weight <= 0:
        raise ValueError("build_decision_matrix: weights must sum to a positive value.")
    norm_weights = {k: max(w, 0.0) / total_weight for k, w in weights.items()}

    df = pd.DataFrame(rows)
    scored = df.copy()

    for criterion in norm_weights:
        if criterion not in df.columns:
            raise KeyError(f"build_decision_matrix: missing criterion '{criterion}' in rows.")
        col = df[criterion].astype(float)
        lo, hi = col.min(), col.max()
        span = hi - lo
        if span == 0:
            normalized = pd.Series([1.0] * len(col), index=col.index)
        else:
            normalized = (col - lo) / span
            if criterion in _COST_CRITERIA:
                normalized = 1.0 - normalized
        scored[f"{criterion}_norm"] = normalized

    scored["Decision Score"] = sum(
        scored[f"{criterion}_norm"] * w for criterion, w in norm_weights.items()
    )
    scored["Rank"] = scored["Decision Score"].rank(ascending=False, method="min").astype(int)

    ordered_cols = [model_key] + [f"{c}_norm" for c in norm_weights] + ["Decision Score", "Rank"]
    remaining_cols = [c for c in scored.columns if c not in ordered_cols]
    scored = scored[ordered_cols + remaining_cols]
    return scored.sort_values("Rank").reset_index(drop=True)
