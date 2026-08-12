import math

import numpy as np
import pytest

from qrepeater_twin.metrics import (
    compute_controller_decision_timing,
    compute_fidelity_statistics,
    compute_temporal_prediction_metrics,
    extract_fidelity_arrays_from_log,
    find_threshold_crossings,
    match_crossings,
)


# ---------------------------------------------------------------------------
# find_threshold_crossings
# ---------------------------------------------------------------------------

def test_find_threshold_crossings_falling_interpolates_correctly():
    # Crosses 0.65 between index 2 (0.8) and index 3 (0.6): linear
    # interpolation places it 75% of the way from 2 to 3.
    series = [1.0, 0.9, 0.8, 0.6, 0.4, 0.3]
    crossings = find_threshold_crossings(series, 0.65, direction="falling")
    assert len(crossings) == 1
    assert crossings[0] == pytest.approx(2.75)


def test_find_threshold_crossings_rising():
    series = [0.3, 0.4, 0.7, 0.9]
    crossings = find_threshold_crossings(series, 0.65, direction="rising")
    assert len(crossings) == 1
    assert 1.0 < crossings[0] < 2.0


def test_find_threshold_crossings_no_crossing_returns_empty():
    series = [0.9, 0.9, 0.9]
    crossings = find_threshold_crossings(series, 0.65, direction="falling")
    assert len(crossings) == 0


def test_find_threshold_crossings_multiple_events():
    # Degrades below 0.65 twice, recovers in between.
    series = [0.9, 0.5, 0.9, 0.4, 0.9]
    crossings = find_threshold_crossings(series, 0.65, direction="falling")
    assert len(crossings) == 2


def test_find_threshold_crossings_invalid_direction_raises():
    with pytest.raises(ValueError):
        find_threshold_crossings([0.9, 0.5], 0.65, direction="sideways")


# ---------------------------------------------------------------------------
# match_crossings
# ---------------------------------------------------------------------------

def test_match_crossings_pairs_nearest_neighbors():
    true_c = [10.0, 50.0]
    pred_c = [11.0, 49.5]
    matches, missed, false_alarms = match_crossings(true_c, pred_c)
    assert len(matches) == 2
    assert not missed and not false_alarms
    pairing = {ti: pi for ti, pi, _ in matches}
    assert pairing[0] == 0 and pairing[1] == 1


def test_match_crossings_respects_max_distance():
    true_c = [10.0, 100.0]
    pred_c = [11.0]  # only close to true[0]
    matches, missed, false_alarms = match_crossings(true_c, pred_c, max_distance=5.0)
    assert len(matches) == 1
    assert missed == [1]
    assert false_alarms == []


def test_match_crossings_empty_inputs():
    matches, missed, false_alarms = match_crossings([], [])
    assert matches == [] and missed == [] and false_alarms == []


def test_match_crossings_all_false_alarms_when_no_true_events():
    matches, missed, false_alarms = match_crossings([], [5.0, 12.0])
    assert matches == []
    assert missed == []
    assert false_alarms == [0, 1]


# ---------------------------------------------------------------------------
# compute_temporal_prediction_metrics
# ---------------------------------------------------------------------------

def _step_degradation(n, drop_at, shift=0.0, slope=0.04):
    t = np.arange(n)
    return np.clip(1.0 - slope * np.maximum(t - drop_at - shift, 0), 0.0, 1.0)


def test_compute_temporal_prediction_metrics_detects_lag():
    y_true = _step_degradation(60, drop_at=25, shift=0.0)
    y_pred = _step_degradation(60, drop_at=25, shift=2.0)  # predictor lags by 2 steps

    result = compute_temporal_prediction_metrics(y_true, y_pred, threshold=0.65, dt=1e-3)

    assert result["n_true_events"] == 1
    assert result["n_pred_events"] == 1
    assert result["n_matched"] == 1
    assert result["mean_timing_error_steps"] == pytest.approx(2.0, abs=1e-6)
    assert result["mean_timing_error_s"] == pytest.approx(2.0e-3, abs=1e-9)
    assert not result["matched_pairs_df"].empty


def test_compute_temporal_prediction_metrics_detects_anticipation():
    y_true = _step_degradation(60, drop_at=25, shift=0.0)
    y_pred = _step_degradation(60, drop_at=25, shift=-3.0)  # predictor anticipates by 3 steps

    result = compute_temporal_prediction_metrics(y_true, y_pred, threshold=0.65)
    assert result["mean_timing_error_steps"] == pytest.approx(-3.0, abs=1e-6)


def test_compute_temporal_prediction_metrics_false_alarm():
    y_true = np.ones(30) * 0.9  # never degrades
    y_pred = np.concatenate([np.ones(15) * 0.9, [0.5], np.ones(14) * 0.9])

    result = compute_temporal_prediction_metrics(y_true, y_pred, threshold=0.65)
    assert result["n_true_events"] == 0
    assert result["n_false_alarms"] == result["n_pred_events"]
    assert result["n_false_alarms"] >= 1


def test_compute_temporal_prediction_metrics_missed_event():
    y_true = _step_degradation(60, drop_at=25, shift=0.0)
    y_pred = np.ones(60) * 0.9  # predictor never flags degradation at all

    result = compute_temporal_prediction_metrics(y_true, y_pred, threshold=0.65)
    assert result["n_true_events"] == 1
    assert result["n_matched"] == 0
    assert result["n_missed_events"] == 1


def test_compute_temporal_prediction_metrics_shape_mismatch_raises():
    with pytest.raises(ValueError):
        compute_temporal_prediction_metrics([0.1, 0.2], [0.1, 0.2, 0.3])


def test_compute_temporal_prediction_metrics_perfect_prediction_zero_error():
    y_true = _step_degradation(60, drop_at=25, shift=0.0)
    result = compute_temporal_prediction_metrics(y_true, y_true, threshold=0.65)
    assert result["mean_timing_error_steps"] == pytest.approx(0.0, abs=1e-9)
    assert result["mean_abs_timing_error_steps"] == pytest.approx(0.0, abs=1e-9)


# ---------------------------------------------------------------------------
# extract_fidelity_arrays_from_log / compute_controller_decision_timing
# ---------------------------------------------------------------------------

def _fake_log(true_vals, pred_vals, threshold=0.65):
    log = []
    for i, (t, p) in enumerate(zip(true_vals, pred_vals)):
        action = "PURIFY" if p >= threshold else "HALT_PURIFICATION"
        log.append({"step": i, "action": action, "true_fidelity": t, "pred_fidelity": p})
    return log


def test_extract_fidelity_arrays_from_log_orders_by_step():
    log = _fake_log([0.9, 0.8, 0.5], [0.85, 0.82, 0.55])
    # Shuffle the log to confirm sorting by 'step' is enforced.
    shuffled = [log[2], log[0], log[1]]
    true_arr, pred_arr = extract_fidelity_arrays_from_log(shuffled)
    assert list(true_arr) == [0.9, 0.8, 0.5]
    assert list(pred_arr) == [0.85, 0.82, 0.55]


def test_extract_fidelity_arrays_from_log_empty_raises():
    with pytest.raises(ValueError):
        extract_fidelity_arrays_from_log([])


def test_extract_fidelity_arrays_from_log_missing_pred_fidelity_raises():
    # Mimics the blind baseline's log entries (PURIFY_BLIND has no pred_fidelity).
    blind_log = [{"step": 0, "action": "PURIFY_BLIND", "true_fidelity": 0.8}]
    with pytest.raises(ValueError):
        extract_fidelity_arrays_from_log(blind_log)


def test_compute_controller_decision_timing_matches_manual_computation():
    y_true = _step_degradation(60, drop_at=25, shift=0.0)
    y_pred = _step_degradation(60, drop_at=25, shift=2.0)
    log = _fake_log(list(y_true), list(y_pred))

    result_from_log = compute_controller_decision_timing(log, threshold=0.65)
    result_direct = compute_temporal_prediction_metrics(y_true, y_pred, threshold=0.65)

    assert result_from_log["mean_timing_error_steps"] == pytest.approx(result_direct["mean_timing_error_steps"])
    assert result_from_log["n_matched"] == result_direct["n_matched"]


# ---------------------------------------------------------------------------
# compute_fidelity_statistics
# ---------------------------------------------------------------------------

def test_compute_fidelity_statistics_basic_shape():
    y_true = _step_degradation(60, drop_at=25, shift=0.0)
    stats = compute_fidelity_statistics(y_true, threshold=0.65)

    assert 0.0 <= stats["mean_fidelity"] <= 1.0
    assert stats["min_fidelity"] <= stats["mean_fidelity"] <= stats["max_fidelity"]
    assert stats["n_degradation_events"] == 1
    assert 0.0 <= stats["pct_below_threshold"] <= 100.0


def test_compute_fidelity_statistics_empty_raises():
    with pytest.raises(ValueError):
        compute_fidelity_statistics([])


def test_compute_fidelity_statistics_constant_series_no_events():
    y_true = np.ones(20) * 0.95
    stats = compute_fidelity_statistics(y_true, threshold=0.65)
    assert stats["n_degradation_events"] == 0
    assert stats["n_recovery_events"] == 0
    assert stats["pct_below_threshold"] == pytest.approx(0.0)
