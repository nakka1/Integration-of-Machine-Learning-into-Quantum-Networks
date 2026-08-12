"""
Decision metrics: the controller's admission decision (HALT vs PURIFY)
evaluated as a binary CLASSIFICATION problem against ground truth (the
photon actually being usable or not), plus the multi-criteria decision
matrix used to rank and select which predictor to deploy.
"""

from __future__ import annotations

from typing import Dict, List

import pandas as pd


# ---------------------------------------------------------------------------
# Admission confusion matrix (derived rates)
# ---------------------------------------------------------------------------

def compute_confusion_metrics(confusion: dict) -> dict:
    """
    Derives precision, recall (sensitivity), specificity, False Positive
    Rate, False Negative Rate, and F1 from the admission confusion matrix
    `{"TP", "FP", "TN", "FN"}` reported by
    `DigitalTwinOrchestrator.run_intelligent` / `run_blind_baseline`.

    The controller's decision is treated as a binary CLASSIFIER: ground
    truth ("good"/"bad" photon) is `true_fidelity >= threshold`; the
    predicted label is the admission DECISION itself
    (`pred_fidelity >= threshold` => PURIFY/"admit"):

        TP : good photon, correctly admitted.
        FP : DEAD photon admitted (F_true < threshold <= F_pred) -- wastes
             a purification attempt (QPU time/shots/energy) on a pair that
             was never going to be useful. This is exactly the error
             `CS_MSELoss.lambda_penalty` penalizes severely. A high FP
             count/rate indicates the controller is "accepting degraded
             states" it should have rejected.
        FN : GOOD photon discarded (F_pred < threshold <= F_true) -- a
             usable pair is thrown away, directly reducing throughput.
             Penalized moderately by `CS_MSELoss.lambda_fn`. A high FN
             count/rate indicates the controller is "rejecting still-usable
             states" -- i.e. being excessively conservative.
        TN : bad photon, correctly halted.

    Returns precision/recall/specificity/FPR/FNR/F1, each `float('nan')`
    when its denominator is zero (e.g. `FPR` is undefined if there were no
    negative/"bad" ground-truth cases at all in the evaluation window).
    """
    tp, fp, tn, fn = confusion["TP"], confusion["FP"], confusion["TN"], confusion["FN"]

    def _safe_div(numerator, denominator):
        return (numerator / denominator) if denominator > 0 else float("nan")

    precision = _safe_div(tp, tp + fp)
    recall = _safe_div(tp, tp + fn)  # a.k.a. sensitivity / True Positive Rate
    specificity = _safe_div(tn, tn + fp)  # True Negative Rate
    fpr = _safe_div(fp, fp + tn)  # dead-photon-admitted rate
    fnr = _safe_div(fn, fn + tp)  # good-photon-discarded rate
    f1 = _safe_div(2 * precision * recall, precision + recall) if (precision == precision and recall == recall) else float("nan")

    return {
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "fpr": fpr,
        "fnr": fnr,
        "f1": f1,
    }


def classify_controller_bias(confusion_rates: dict, fpr_high: float = 0.15, fnr_high: float = 0.15) -> str:
    """
    Human-readable qualitative verdict from `compute_confusion_metrics`'s
    output, answering directly: is the controller (a) accepting degraded
    states, (b) rejecting still-usable states, or (c) excessively
    conservative overall?

    Thresholds `fpr_high`/`fnr_high` (default 15%) are deliberately
    simple, inspectable cutoffs -- NOT a claim of statistical
    significance -- meant to turn a table of numbers into an immediate,
    qualitative reading; always inspect the raw `fpr`/`fnr` values
    alongside this verdict for the actual magnitude.
    """
    fpr, fnr = confusion_rates["fpr"], confusion_rates["fnr"]
    fpr_flag = (fpr == fpr) and (fpr > fpr_high)
    fnr_flag = (fnr == fnr) and (fnr > fnr_high)

    if fpr_flag and fnr_flag:
        return (f"Controller shows BOTH a high false-positive rate ({fpr*100:.1f}% > {fpr_high*100:.0f}%, "
                f"accepting degraded states) AND a high false-negative rate ({fnr*100:.1f}% > {fnr_high*100:.0f}%, "
                f"rejecting still-usable states) -- decision quality is poor in both directions.")
    if fpr_flag:
        return (f"Controller is ACCEPTING DEGRADED STATES too often: false-positive rate "
                f"{fpr*100:.1f}% exceeds the {fpr_high*100:.0f}% threshold (dead photons are being "
                f"admitted, wasting QPU time on doomed purification attempts).")
    if fnr_flag:
        return (f"Controller is EXCESSIVELY CONSERVATIVE: false-negative rate {fnr*100:.1f}% exceeds "
                f"the {fnr_high*100:.0f}% threshold (still-usable photons are being rejected, "
                f"unnecessarily reducing throughput).")
    return (f"Controller decision quality is within the inspected bounds: false-positive rate "
            f"{fpr*100:.1f}% and false-negative rate {fnr*100:.1f}% are both <= {max(fpr_high, fnr_high)*100:.0f}%.")


# ---------------------------------------------------------------------------
# Multi-criteria decision matrix
# ---------------------------------------------------------------------------

# Criteria where a LOWER raw value is better (inverted before weighting).
# `latency_ratio_c` (C_latencia = tau_inf / T2) replaces a raw
# `inference_latency_ms` criterion -- see `performance.compute_latency_ratio`.
_COST_CRITERIA = {"latency_ratio_c", "total_energy_j"}


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
        `energy_saved_pct`, `latency_ratio_c`).
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
