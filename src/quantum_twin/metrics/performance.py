"""
Performance metrics: throughput, inference/classical latency expressed as
a dimensionless physical ratio against the qubit's own coherence time, and
an illustrative energy accounting comparing a predictive controller
against the blind/reactive baseline.
"""

from __future__ import annotations

from ..config import EnergyConfig


# ---------------------------------------------------------------------------
# Dimensionless temporal-scale ratio: C_latencia = tau_inf / T2
# ---------------------------------------------------------------------------

def compute_latency_ratio(avg_classical_latency_s: float, T2: float) -> float:
    """
    C_latencia = tau_inf / T2 -- the classical inference latency expressed
    as a FRACTION of the qubit's T2 coherence time, replacing a raw
    millisecond comparison with a dimensionless physical constraint.

    A millisecond figure alone says nothing about whether the classical
    decision-making overhead actually matters to the quantum hardware it
    gates: 100 microseconds is negligible for a T2 of seconds, and
    catastrophic for a T2 of microseconds. C_latencia << 1 means the
    forward pass is effectively "free" relative to decoherence; C_latencia
    approaching or exceeding 1 means the qubit has already substantially
    (or fully) decohered by the time the admission decision is made,
    undermining the entire premise of predictive admission control.

    Raises `ValueError` if `T2 <= 0` (undefined ratio).
    """
    if T2 <= 0:
        raise ValueError(f"compute_latency_ratio: T2 must be positive, got {T2!r}.")
    return avg_classical_latency_s / T2


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
