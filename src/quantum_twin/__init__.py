"""
quantum_twin
==============

Digital Twin of a Quantum Repeater with a predictive admission controller
(EdgeLSTM) and a Pareto Frontier sweep over the False Positive penalty of
the CS_MSELoss -- plus a cross-architecture comparison against LSTM+MSE,
Random Forest, XGBoost, Transformer, and naive/oracle (Persistence,
Moving Average, Oracle) baselines, a full evaluation of the EdgeLSTM's
own predictive capability (MAE/RMSE/R^2 AND temporal threshold-crossing
timing analysis), a 2x2 factorial ablation study ({EdgeLSTM,
StandardLSTM} x {MSE, CS_MSELoss}), an admission confusion matrix
(FP/FN/TP/TN) treated as a binary classification problem, the
dimensionless latency ratio C_latencia = tau_inf/T2, throughput, QPU
economy, energy accounting, a ranked multi-criteria decision matrix, an
automatic +/-10% decision-weight sensitivity analysis, paired
significance testing with Holm-Bonferroni correction, walk-forward
(rolling-origin) temporal cross-validation, full CPU/GPU determinism,
plotting helpers, LaTeX table export, and an automatic
experiment-tracking pipeline that records every run's
configuration/tables/figures to disk.

This package decomposes the original prototype (a single monolithic
notebook) into independent, testable modules, suitable for a reproducible
GitHub repository:

    channel_simulator.py    -> WDMChannelSimulator (synthetic data generation)
    models.py                 -> EdgeLSTM, StandardLSTM, CS_MSELoss, train_edge_lstm
    timing.py                   -> InferenceTimer (hardware-accurate profiling, CUDA Events)
    quantum_node.py               -> QuantumRepeaterNode (virtual quantum dataplane via Qiskit Aer)
    orchestrator.py                 -> DigitalTwinOrchestrator (intelligent/blind loops, confusion matrix)
    pareto_sweep.py                   -> run_pareto_sweep (statistically robust, multi-seed)
    baselines.py                        -> LSTM+MSE, Random Forest, XGBoost, Transformer,
                                            Persistence/MovingAverage/Oracle predictors
    metrics/                              -> split into prediction.py (MAE/RMSE/R^2 + temporal
                                              crossing-timing analysis), quantum.py (QPU economy,
                                              fidelity statistics), performance.py (throughput,
                                              C_latencia, energy), decision.py (confusion-matrix
                                              classification metrics, decision matrix)
    model_comparison.py                     -> run_model_comparison (cross-architecture, multi-seed)
    ablation.py                               -> run_ablation_study (2x2 factorial: architecture x loss)
    sensitivity.py                              -> run_weight_sensitivity_analysis (+/-10% weight robustness)
    statistics_tests.py                           -> paired t-test/Wilcoxon + Holm-Bonferroni + CIs
    walk_forward.py                                 -> run_walk_forward_evaluation (rolling-origin CV)
    reproducibility.py                                -> seed_everything, set_full_determinism
    latex_export.py                                     -> dataframe_to_latex, export_all_results_to_latex
    plotting.py                                           -> matplotlib chart generators for every result table
    experiment_tracking.py                                  -> ExperimentRun + per-experiment-type trackers
    mlops.py                                                  -> MLflowTracker (optional MLflow-backed tracking)
    config.py                                                 -> configuration dataclasses
    cli.py                                                      -> command-line entry point (main)
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
    PersistencePredictor,
    MovingAveragePredictor,
    OraclePredictor,
    train_lstm_mse,
    train_random_forest,
    train_xgboost,
    train_transformer,
    build_persistence_baseline,
    build_moving_average_baseline,
    build_oracle_baseline,
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
from .statistics_tests import (
    compute_confidence_interval,
    paired_ttest,
    wilcoxon_test,
    holm_bonferroni_correction,
    compare_models_statistically,
)
from .walk_forward import generate_walk_forward_splits, run_walk_forward_evaluation
from .reproducibility import seed_everything, set_full_determinism
from .latex_export import dataframe_to_latex, export_all_results_to_latex
from .mlops import MLFLOW_AVAILABLE, MLflowTracker
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
    WalkForwardConfig,
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
    "PersistencePredictor",
    "MovingAveragePredictor",
    "OraclePredictor",
    "train_lstm_mse",
    "train_random_forest",
    "train_xgboost",
    "train_transformer",
    "build_persistence_baseline",
    "build_moving_average_baseline",
    "build_oracle_baseline",
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
    "compute_confidence_interval",
    "paired_ttest",
    "wilcoxon_test",
    "holm_bonferroni_correction",
    "compare_models_statistically",
    "generate_walk_forward_splits",
    "run_walk_forward_evaluation",
    "seed_everything",
    "set_full_determinism",
    "dataframe_to_latex",
    "export_all_results_to_latex",
    "MLFLOW_AVAILABLE",
    "MLflowTracker",
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
    "WalkForwardConfig",
]

__version__ = "4.0.0"
