# `quantum_twin.metrics`

The `metrics` subpackage is split into four single-responsibility
modules; every name below is also re-exported from `quantum_twin.metrics`
directly (`from quantum_twin.metrics import compute_regression_metrics`
keeps working regardless of which submodule an object actually lives in).

## `quantum_twin.metrics.prediction`

Regression quality (MAE/RMSE/R²) and temporal (event-based)
threshold-crossing timing analysis.

::: quantum_twin.metrics.prediction

## `quantum_twin.metrics.quantum`

QPU-time economy and true-fidelity-trajectory statistics.

::: quantum_twin.metrics.quantum

## `quantum_twin.metrics.performance`

Throughput, the dimensionless latency ratio C_latencia = tau_inf/T2, and
energy accounting.

::: quantum_twin.metrics.performance

## `quantum_twin.metrics.decision`

Admission confusion-matrix classification metrics and the multi-criteria
decision matrix.

::: quantum_twin.metrics.decision
