"""
Component 6 -- Derived operational metrics, now split into four
independent, single-responsibility submodules for maintainability and
scientific clarity:

    - `prediction.py`  : pure ML regression quality of F_hat(t)
                          (MAE/RMSE/R^2) AND temporal (event-based)
                          analysis of threshold-crossing timing --
                          "when" the predictor detects degradation
                          relative to when it actually happens.
    - `quantum.py`      : QPU-time economy relative to the blind baseline,
                          and descriptive statistics of the true fidelity
                          trajectory itself (degradation event counts,
                          etc.).
    - `performance.py`  : throughput, the dimensionless latency ratio
                          C_latencia = tau_inf / T2, and illustrative
                          energy accounting.
    - `decision.py`     : the controller's admission decision evaluated
                          as a binary classifier (confusion matrix ->
                          precision/recall/F1), plus the multi-criteria
                          decision matrix used to rank predictor models.

This `__init__.py` re-exports every public name from all four submodules
under `qrepeater_twin.metrics`, so existing code (`from .metrics import
compute_regression_metrics, ...`) keeps working completely unmodified --
the split is purely an internal reorganization, not a breaking API
change.
"""

from .prediction import (
    compute_regression_metrics,
    evaluate_predictor_regression,
    predict_sequence,
    find_threshold_crossings,
    match_crossings,
    compute_temporal_prediction_metrics,
    extract_fidelity_arrays_from_log,
    compute_controller_decision_timing,
)
from .quantum import (
    compute_qpu_economy,
    compute_fidelity_statistics,
)
from .performance import (
    compute_latency_ratio,
    compute_throughput,
    compute_energy_report,
)
from .decision import (
    compute_confusion_metrics,
    classify_controller_bias,
    build_decision_matrix,
)

__all__ = [
    # prediction.py
    "compute_regression_metrics",
    "evaluate_predictor_regression",
    "predict_sequence",
    "find_threshold_crossings",
    "match_crossings",
    "compute_temporal_prediction_metrics",
    "extract_fidelity_arrays_from_log",
    "compute_controller_decision_timing",
    # quantum.py
    "compute_qpu_economy",
    "compute_fidelity_statistics",
    # performance.py
    "compute_latency_ratio",
    "compute_throughput",
    "compute_energy_report",
    # decision.py
    "compute_confusion_metrics",
    "classify_controller_bias",
    "build_decision_matrix",
]
