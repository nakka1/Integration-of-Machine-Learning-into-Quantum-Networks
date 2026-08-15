"""
Prediction metrics: pure Machine-Learning regression quality of the
F_hat(t) predictor (MAE / RMSE / R^2), plus temporal analysis of *when*
the predictor detects channel degradation relative to when it actually
happens.

Everything in this module is computed on the predictor's raw output
sequence, entirely BEFORE and decoupled from any admission-control logic
or the quantum dataplane -- these are metrics of the forecaster itself,
not of the final HALT/PURIFY decision (that is `metrics.decision`'s job).

Two complementary views of predictive quality:

    1. Point-wise regression accuracy (MAE/RMSE/R^2) -- "how close is
       F_hat(t) to F(t), on average, across the whole test window?"
    2. Temporal (event-based) accuracy -- "when the channel actually
       degrades below the admission threshold, does the predictor see it
       coming at the right TIME, or does it lag behind / hallucinate
       degradation events that never happen?" A model can have excellent
       point-wise MAE yet still be systematically early or late at the
       one moment that matters operationally: the threshold crossing that
       triggers a HALT decision.
"""

from __future__ import annotations

from typing import List, Protocol, Sequence, Tuple, Union

import numpy as np
import pandas as pd
import torch

# Every metric function below accepts EITHER a torch.Tensor (as produced
# directly by a model's forward pass / the dataset's y_test) or plain
# array-like data (numpy array, list, tuple) -- this alias documents that
# shared contract once instead of repeating the Union at every call site.
ArrayLike = Union[torch.Tensor, np.ndarray, Sequence[float]]


class PredictorLike(Protocol):
    """
    The minimal duck-typed contract every predictor in this project
    satisfies -- a trained `torch.nn.Module` (`EdgeLSTM`/`StandardLSTM`/
    `TinyTransformer`), a `baselines._SklearnRegressorAdapter`, or one of
    the naive/oracle `_NaiveSequentialPredictorBase` subclasses
    (`PersistencePredictor`, `MovingAveragePredictor`, `OraclePredictor`).
    Declared as a `Protocol` (structural typing) rather than a shared base
    class, since these predictors are intentionally NOT related by
    inheritance -- only by this common `.eval()`/`__call__` interface.
    """

    def eval(self) -> "PredictorLike": ...
    def __call__(self, x: torch.Tensor) -> torch.Tensor: ...


# ---------------------------------------------------------------------------
# Pure Machine-Learning regression metrics (MAE / RMSE / R^2)
# ---------------------------------------------------------------------------

def _to_numpy(x: ArrayLike) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def compute_regression_metrics(y_true: ArrayLike, y_pred: ArrayLike) -> dict:
    """
    Pure regression-quality metrics of the F_hat(t) predictor: Mean
    Absolute Error (MAE), Root Mean Squared Error (RMSE), and the
    coefficient of determination (R^2).

    Computed BEFORE any admission-control logic (threshold comparison,
    HALT/PURIFY decision) is applied -- this is the predictor evaluated as
    a plain regressor against the true fidelity trajectory, independent of
    the digital-twin/quantum simulation loop entirely. `y_true`/`y_pred`
    accept `torch.Tensor` or array-like of any shape (flattened
    internally); a torch tensor with `requires_grad=True` is safely
    detached first.

    R^2 is defined as `1 - SS_res / SS_tot` and is left as `float('nan')`
    on the degenerate case `SS_tot == 0` (constant `y_true`), rather than
    silently reporting a spurious/undefined value.
    """
    y_true_np = _to_numpy(y_true).ravel().astype(float)
    y_pred_np = _to_numpy(y_pred).ravel().astype(float)
    if y_true_np.shape != y_pred_np.shape:
        raise ValueError(
            f"compute_regression_metrics: shape mismatch after flattening "
            f"({y_true_np.shape} vs {y_pred_np.shape})."
        )

    errors = y_pred_np - y_true_np
    mae = float(np.mean(np.abs(errors)))
    rmse = float(np.sqrt(np.mean(errors ** 2)))

    ss_res = float(np.sum(errors ** 2))
    ss_tot = float(np.sum((y_true_np - y_true_np.mean()) ** 2))
    r2 = (1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan")

    return {"mae": mae, "rmse": rmse, "r2": r2}


def evaluate_predictor_regression(model: PredictorLike, X_test: torch.Tensor, y_test: torch.Tensor,
                                   device: torch.device | None = None) -> dict:
    """
    Runs ONE full-batch forward pass of `model` over `X_test` (no
    admission logic, no quantum dataplane involved) and reports
    `compute_regression_metrics` against `y_test`.

    Works uniformly for a trained `EdgeLSTM`/`StandardLSTM`/`TinyTransformer`
    (`torch.nn.Module`, moved to `device` and run under
    `torch.no_grad()`) and for the tree-ensemble baselines wrapped by
    `baselines._SklearnRegressorAdapter` (which ignores `device` and
    always returns a CPU tensor) -- both expose the same `.eval()` /
    callable contract, so this function never needs to know which kind of
    predictor it was handed.
    """
    model.eval()
    X = X_test.to(device) if device is not None else X_test
    with torch.no_grad():
        y_pred = model(X)
    return compute_regression_metrics(y_test, y_pred)


def predict_sequence(model: PredictorLike, X_test: torch.Tensor, device: torch.device | None = None) -> np.ndarray:
    """
    Runs ONE full-batch forward pass of `model` over `X_test` and returns
    the flattened prediction sequence as a plain 1-D numpy array -- the
    shared entry point `compute_temporal_prediction_metrics` and
    `evaluate_predictor_regression` both build on, so the model is only
    ever invoked once per evaluation.
    """
    model.eval()
    X = X_test.to(device) if device is not None else X_test
    with torch.no_grad():
        y_pred = model(X)
    return _to_numpy(y_pred).ravel().astype(float)


# ---------------------------------------------------------------------------
# Temporal (event-based) prediction analysis: threshold-crossing timing
# ---------------------------------------------------------------------------

def find_threshold_crossings(series: ArrayLike, threshold: float, direction: str = "falling") -> np.ndarray:
    """
    Locates every point where `series` crosses `threshold`, returning
    FRACTIONAL (linearly-interpolated) time indices rather than only the
    nearest integer sample -- e.g. a crossing found between samples 12 and
    13, three-quarters of the way from 12 to 13, is reported as `12.75`.
    This sub-sample precision matters because the "instant of degradation"
    is a continuous physical event that the discrete sampling grid only
    approximates; rounding to the nearest sample would inject up to +/-0.5
    samples of pure discretization error into every timing comparison
    below.

    `direction`:
        - "falling" : `series[i] >= threshold` then `series[i+1] < threshold`
                       (channel degrading below the admission threshold --
                       the event of interest for HALT/PURIFY timing).
        - "rising"  : `series[i] < threshold` then `series[i+1] >= threshold`
                       (channel recovering above the threshold).

    Returns a 1-D numpy array of fractional indices, sorted ascending
    (empty if `series` never crosses `threshold` in the requested
    direction).
    """
    if direction not in ("falling", "rising"):
        raise ValueError(f"find_threshold_crossings: direction must be 'falling' or 'rising', got {direction!r}.")

    s = np.asarray(series, dtype=float).ravel()
    crossings = []
    for i in range(len(s) - 1):
        a, b = s[i], s[i + 1]
        if direction == "falling":
            crossed = (a >= threshold) and (b < threshold)
        else:
            crossed = (a < threshold) and (b >= threshold)
        if not crossed:
            continue
        denom = b - a
        frac = (threshold - a) / denom if denom != 0 else 0.0
        frac = min(max(frac, 0.0), 1.0)  # numerical safety
        crossings.append(i + frac)

    return np.asarray(crossings, dtype=float)


def _interpolate_at(series, frac_index: float) -> float:
    """Linearly interpolates `series` at a (possibly fractional) index."""
    s = np.asarray(series, dtype=float).ravel()
    i0 = int(np.floor(frac_index))
    i1 = min(i0 + 1, len(s) - 1)
    i0 = min(max(i0, 0), len(s) - 1)
    w = frac_index - i0
    return float(s[i0] * (1.0 - w) + s[i1] * w)


def match_crossings(true_crossings: Sequence[float], pred_crossings: Sequence[float],
                     max_distance: float | None = None) -> Tuple[List[Tuple[int, int, float]], List[int], List[int]]:
    """
    Greedy nearest-neighbor 1-to-1 matching between two sorted lists of
    crossing positions (fractional time indices).

    Repeatedly picks the globally closest still-unmatched (true, pred)
    pair, ties broken by array order, until no pair remains within
    `max_distance` (or all crossings on one side are exhausted). This is a
    standard greedy approximation to the assignment problem -- adequate
    here because crossing events are typically sparse and well-separated
    relative to the sampling grid, so the greedy and optimal (Hungarian)
    assignments coincide in the overwhelming majority of cases, at a
    fraction of the implementation complexity.

    Returns
    -------
    matches : list[(true_idx, pred_idx, distance)]
        Indices into `true_crossings`/`pred_crossings` (not the crossing
        values themselves), plus the matched absolute time distance.
    missed_true_idx : list[int]
        Indices of true crossings with NO matching predicted crossing
        within `max_distance` -- degradation events the predictor missed
        entirely.
    false_pred_idx : list[int]
        Indices of predicted crossings with no corresponding true
        crossing -- degradation events the predictor "hallucinated".
    """
    true_crossings = list(true_crossings)
    pred_crossings = list(pred_crossings)

    candidates = []
    for ti, t in enumerate(true_crossings):
        for pi, p in enumerate(pred_crossings):
            dist = abs(p - t)
            if max_distance is not None and dist > max_distance:
                continue
            candidates.append((dist, ti, pi))
    candidates.sort(key=lambda c: c[0])

    matched_true, matched_pred = set(), set()
    matches: List[Tuple[int, int, float]] = []
    for dist, ti, pi in candidates:
        if ti in matched_true or pi in matched_pred:
            continue
        matched_true.add(ti)
        matched_pred.add(pi)
        matches.append((ti, pi, dist))

    missed_true_idx = [i for i in range(len(true_crossings)) if i not in matched_true]
    false_pred_idx = [i for i in range(len(pred_crossings)) if i not in matched_pred]
    return matches, missed_true_idx, false_pred_idx


def compute_temporal_prediction_metrics(y_true: ArrayLike, y_pred: ArrayLike, threshold: float = 0.65,
                                         dt: float | None = None, max_match_distance: float | None = None,
                                         direction: str = "falling") -> dict:
    """
    Event-based temporal accuracy of the predictor's threshold crossings:
    does F_hat(t) cross below `threshold` at the same TIME the true F(t)
    does?

    For each matched (true, predicted) crossing pair, the signed timing
    error is:

        timing_error = predicted_crossing_index - true_crossing_index

    with the sign convention:
        - timing_error > 0 : the predictor crosses LATE -- it detects
          degradation *after* it has already happened (a lagging /
          delayed controller decision: HALT arrives too late to avoid
          admitting an already-dead photon).
        - timing_error < 0 : the predictor crosses EARLY -- it anticipates
          degradation before it actually occurs (a controller that HALTs
          too conservatively, ahead of the real event).

    Also reports, per matched pair, the VALUE-space error at the true
    crossing instant: `pred_value_at_true_crossing - threshold` (positive
    => the predictor still thought the channel was fine exactly when it
    wasn't; negative => the predictor had already flagged degradation by
    then), a complementary view to the purely temporal error above.

    Parameters
    ----------
    y_true, y_pred : array-like or torch.Tensor
        The TRUE and PREDICTED fidelity sequences, in chronological order
        (one point per test-window step -- e.g. `y_test`/`predict_sequence(...)`).
    threshold : float
        Admission threshold (same value used by `CS_MSELoss`/the
        orchestrator).
    dt : float, optional
        Seconds per step (e.g. `ComparisonConfig.cycle_time_s`). When
        given, every "_steps" quantity below also gets a "_s" (seconds)
        counterpart.
    max_match_distance : float, optional
        Maximum allowed distance (in fractional steps) for a (true, pred)
        crossing pair to be considered a match; unmatched crossings beyond
        this are reported as missed events / false alarms instead of
        being forced into a nonsensical pairing. `None` (default) allows
        any distance.
    direction : {"falling", "rising"}
        Which crossing direction to analyze (default "falling" --
        degradation events, the ones that drive HALT decisions).

    Returns
    -------
    dict with:
        n_true_events, n_pred_events, n_matched, n_missed_events, n_false_alarms
        mean_timing_error_steps, std_timing_error_steps, median_timing_error_steps
        mean_abs_timing_error_steps
        mean_value_error_at_crossing   (predicted-fidelity gap at the true crossing instant)
        timing_errors_steps            : raw list, one per matched pair (for histograms)
        matched_pairs_df                : pd.DataFrame, one row per matched pair
                                           (true_step, pred_step, timing_error_steps,
                                           value_error_at_crossing[, *_s columns])
        (all "_steps" fields get "_s" seconds counterparts when `dt` is given)
    """
    y_true_np = _to_numpy(y_true).ravel().astype(float)
    y_pred_np = _to_numpy(y_pred).ravel().astype(float)
    if y_true_np.shape != y_pred_np.shape:
        raise ValueError(
            f"compute_temporal_prediction_metrics: shape mismatch after flattening "
            f"({y_true_np.shape} vs {y_pred_np.shape})."
        )

    true_crossings = find_threshold_crossings(y_true_np, threshold, direction=direction)
    pred_crossings = find_threshold_crossings(y_pred_np, threshold, direction=direction)

    matches, missed_true_idx, false_pred_idx = match_crossings(
        true_crossings, pred_crossings, max_distance=max_match_distance,
    )

    rows = []
    for ti, pi, _dist in matches:
        t_cross, p_cross = true_crossings[ti], pred_crossings[pi]
        timing_error = p_cross - t_cross
        value_error = _interpolate_at(y_pred_np, t_cross) - threshold
        row = {
            "true_step": t_cross,
            "pred_step": p_cross,
            "timing_error_steps": timing_error,
            "value_error_at_crossing": value_error,
        }
        if dt is not None:
            row["timing_error_s"] = timing_error * dt
        rows.append(row)

    matched_pairs_df = pd.DataFrame(rows, columns=(
        ["true_step", "pred_step", "timing_error_steps", "value_error_at_crossing"]
        + (["timing_error_s"] if dt is not None else [])
    ))

    timing_errors = [r["timing_error_steps"] for r in rows]
    value_errors = [r["value_error_at_crossing"] for r in rows]

    result = {
        "n_true_events": len(true_crossings),
        "n_pred_events": len(pred_crossings),
        "n_matched": len(matches),
        "n_missed_events": len(missed_true_idx),
        "n_false_alarms": len(false_pred_idx),
        "mean_timing_error_steps": float(np.mean(timing_errors)) if timing_errors else float("nan"),
        "std_timing_error_steps": float(np.std(timing_errors)) if timing_errors else float("nan"),
        "median_timing_error_steps": float(np.median(timing_errors)) if timing_errors else float("nan"),
        "mean_abs_timing_error_steps": float(np.mean(np.abs(timing_errors))) if timing_errors else float("nan"),
        "mean_value_error_at_crossing": float(np.mean(value_errors)) if value_errors else float("nan"),
        "timing_errors_steps": timing_errors,
        "matched_pairs_df": matched_pairs_df,
    }
    if dt is not None:
        result["mean_timing_error_s"] = result["mean_timing_error_steps"] * dt
        result["std_timing_error_s"] = result["std_timing_error_steps"] * dt
        result["mean_abs_timing_error_s"] = result["mean_abs_timing_error_steps"] * dt

    return result


def extract_fidelity_arrays_from_log(log: List[dict]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extracts chronologically-ordered `(true_fidelity, pred_fidelity)`
    arrays from `DigitalTwinOrchestrator.run_intelligent`'s returned log
    (`orchestrator.log`, or the `results` list it builds internally).

    Every entry in that log -- whether the action was `HALT_PURIFICATION`
    or `PURIFY` -- carries both `pred_fidelity` and `true_fidelity` (the
    admission decision only gates whether the QUANTUM circuit runs, not
    whether the classical prediction is recorded), so this always
    recovers the full, unbroken prediction timeline. Raises `ValueError`
    if any entry lacks `pred_fidelity` -- this happens for
    `run_blind_baseline`'s log (`PURIFY_BLIND` actions), which never runs
    a predictor at all and therefore has no prediction timeline to
    analyze.
    """
    if not log:
        raise ValueError("extract_fidelity_arrays_from_log: log is empty.")
    ordered = sorted(log, key=lambda r: r["step"])
    if any("pred_fidelity" not in r for r in ordered):
        raise ValueError(
            "extract_fidelity_arrays_from_log: at least one log entry has no 'pred_fidelity' "
            "(this happens for the blind baseline's log, which never runs a predictor -- "
            "temporal prediction analysis only applies to DigitalTwinOrchestrator.run_intelligent)."
        )
    true_arr = np.array([r["true_fidelity"] for r in ordered], dtype=float)
    pred_arr = np.array([r["pred_fidelity"] for r in ordered], dtype=float)
    return true_arr, pred_arr


def compute_controller_decision_timing(log: List[dict], threshold: float = 0.65,
                                        dt: float | None = None,
                                        max_match_distance: float | None = None) -> dict:
    """
    Convenience wrapper: extracts the true/predicted fidelity timeline
    from a `DigitalTwinOrchestrator.run_intelligent` log via
    `extract_fidelity_arrays_from_log`, then runs
    `compute_temporal_prediction_metrics` on it.

    Because the admission decision (HALT vs PURIFY) is itself just
    `pred_fidelity >= threshold`, a "falling" crossing of `pred_fidelity`
    below `threshold` IS the controller's HALT decision boundary -- so
    `timing_error_steps > 0` here means the controller's HALT decision
    arrived LATE (after the channel had already degraded: it anticipated
    and correctly, or belatedly, admitted a bad cycle) and
    `timing_error_steps < 0` means the controller HALTed EARLY (ahead of
    the actual degradation event -- an anticipatory / conservative
    decision). This directly answers "anticipation or delay of the
    controller's decision" in terms already tied to the concrete
    HALT/PURIFY log, without requiring the caller to re-derive prediction
    arrays by hand.
    """
    true_arr, pred_arr = extract_fidelity_arrays_from_log(log)
    return compute_temporal_prediction_metrics(
        true_arr, pred_arr, threshold=threshold, dt=dt,
        max_match_distance=max_match_distance, direction="falling",
    )
