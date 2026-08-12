"""
qrepeater_twin
==============

Digital Twin of a Quantum Repeater with a predictive admission controller
(EdgeLSTM) and a Pareto Frontier sweep over the False Positive penalty of
the CS_MSELoss -- plus a cross-architecture comparison against LSTM+MSE,
Random Forest, XGBoost, and Transformer baselines, a full evaluation of
the EdgeLSTM's own predictive capability (MAE/RMSE/R^2 AND temporal
threshold-crossing timing analysis), a 2x2 factorial ablation study
({EdgeLSTM, StandardLSTM} x {MSE, CS_MSELoss}), an admission confusion
matrix (FP/FN/TP/TN) treated as a binary classification problem, the
dimensionless latency ratio C_latencia = tau_inf/T2, throughput, QPU
economy, energy accounting, a ranked multi-criteria decision matrix, an
automatic +/-10% decision-weight sensitivity analysis, plotting helpers,
and an automatic experiment-tracking pipeline that records every run's
configuration/tables/figures to disk.

This package decomposes the original prototype (a single monolithic
notebook) into independent, testable modules, suitable for a reproducible
GitHub repository:

    channel_simulator.py   -> WDMChannelSimulator (synthetic data generation)
    models.py                -> EdgeLSTM, StandardLSTM, CS_MSELoss, train_edge_lstm
    timing.py                  -> InferenceTimer (hardware-accurate profiling, CUDA Events)
    quantum_node.py              -> QuantumRepeaterNode (virtual quantum dataplane via Qiskit Aer)
    orchestrator.py                -> DigitalTwinOrchestrator (intelligent/blind loops, confusion matrix)
    pareto_sweep.py                  -> run_pareto_sweep (statistically robust, multi-seed)
    baselines.py                       -> LSTM+MSE, Random Forest, XGBoost, Transformer predictors
    metrics/                             -> split into prediction.py (MAE/RMSE/R^2 + temporal
                                             crossing-timing analysis), quantum.py (QPU economy,
                                             fidelity statistics), performance.py (throughput,
                                             C_latencia, energy), decision.py (confusion-matrix
                                             classification metrics, decision matrix)
    model_comparison.py                    -> run_model_comparison (cross-architecture, multi-seed)
    ablation.py                              -> run_ablation_study (2x2 factorial: architecture x loss)
    sensitivity.py                             -> run_weight_sensitivity_analysis (+/-10% weight robustness)
    plotting.py                                  -> matplotlib chart generators for every result table
    experiment_tracking.py                         -> ExperimentRun + per-experiment-type trackers
    config.py                                        -> configuration dataclasses
    cli.py                                             -> command-line entry point (main)
"""

from .channel_simulator import WDMChannelSimulator
from .models import EdgeLSTM, StandardLSTM, CS_MSELoss, train_edge_lstm
from .timing import InferenceTimer
from .quantum_node import QuantumRepeaterNode
from .orchestrator import DigitalTwinOrchestrator
from .pareto_sweep import run_pareto_sweep
from .baselines import (
    TinyTransformer,
    RandomForestFidelityModel,
    XGBoostFidelityModel,
    train_lstm_mse,
    train_random_forest,
    train_xgboost,
    train_transformer,
)
from .metrics import (
    compute_regression_metrics,
    evaluate_predictor_regression,
    predict_sequence,
    find_threshold_crossings,
    match_crossings,
    compute_temporal_prediction_metrics,
    extract_fidelity_arrays_from_log,
    compute_controller_decision_timing,
    compute_qpu_economy,
    compute_fidelity_statistics,
    compute_confusion_metrics,
    classify_controller_bias,
    compute_latency_ratio,
    compute_throughput,
    compute_energy_report,
    build_decision_matrix,
)
from .model_comparison import run_model_comparison
from .ablation import run_ablation_study
from .sensitivity import run_weight_sensitivity_analysis, summarize_robustness
from .experiment_tracking import (
    ExperimentRun,
    track_pareto_sweep_experiment,
    track_model_comparison_experiment,
    track_ablation_experiment,
)
from .config import (
    SimConfig,
    TrainConfig,
    QuantumConfig,
    SweepConfig,
    BaselineConfig,
    EnergyConfig,
    ComparisonConfig,
    AblationConfig,
)

__all__ = [
    "WDMChannelSimulator",
    "EdgeLSTM",
    "StandardLSTM",
    "CS_MSELoss",
    "train_edge_lstm",
    "InferenceTimer",
    "QuantumRepeaterNode",
    "DigitalTwinOrchestrator",
    "run_pareto_sweep",
    "TinyTransformer",
    "RandomForestFidelityModel",
    "XGBoostFidelityModel",
    "train_lstm_mse",
    "train_random_forest",
    "train_xgboost",
    "train_transformer",
    "compute_regression_metrics",
    "evaluate_predictor_regression",
    "predict_sequence",
    "find_threshold_crossings",
    "match_crossings",
    "compute_temporal_prediction_metrics",
    "extract_fidelity_arrays_from_log",
    "compute_controller_decision_timing",
    "compute_qpu_economy",
    "compute_fidelity_statistics",
    "compute_confusion_metrics",
    "classify_controller_bias",
    "compute_latency_ratio",
    "compute_throughput",
    "compute_energy_report",
    "build_decision_matrix",
    "run_model_comparison",
    "run_ablation_study",
    "run_weight_sensitivity_analysis",
    "summarize_robustness",
    "ExperimentRun",
    "track_pareto_sweep_experiment",
    "track_model_comparison_experiment",
    "track_ablation_experiment",
    "SimConfig",
    "TrainConfig",
    "QuantumConfig",
    "SweepConfig",
    "BaselineConfig",
    "EnergyConfig",
    "ComparisonConfig",
    "AblationConfig",
]

__version__ = "3.2.0"
