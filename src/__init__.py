"""
qrepeater_twin
==============

Digital Twin of a Quantum Repeater with a predictive admission controller
(EdgeLSTM) and a Pareto Frontier sweep over the False Positive penalty of
the CS_MSELoss -- plus a cross-architecture comparison against LSTM+MSE,
Random Forest, XGBoost, and Transformer baselines, with throughput, QPU
economy, and energy accounting, and a ranked multi-criteria decision
matrix.

This package decomposes the original prototype (a single monolithic
notebook) into independent, testable modules, suitable for a reproducible
GitHub repository:

    channel_simulator.py  -> WDMChannelSimulator (synthetic data generation)
    models.py               -> EdgeLSTM, CS_MSELoss, train_edge_lstm
    timing.py                 -> InferenceTimer (hardware-accurate profiling, CUDA Events)
    quantum_node.py            -> QuantumRepeaterNode (virtual quantum dataplane via Qiskit Aer)
    orchestrator.py             -> DigitalTwinOrchestrator (intelligent/blind loops)
    pareto_sweep.py               -> run_pareto_sweep (statistically robust, multi-seed)
    baselines.py                   -> LSTM+MSE, Random Forest, XGBoost, Transformer predictors
    metrics.py                      -> throughput, QPU economy, energy, decision matrix
    model_comparison.py              -> run_model_comparison (cross-architecture, multi-seed)
    config.py                         -> configuration dataclasses
    cli.py                              -> command-line entry point (main)
"""

from .channel_simulator import WDMChannelSimulator
from .models import EdgeLSTM, CS_MSELoss, train_edge_lstm
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
    compute_throughput,
    compute_qpu_economy,
    compute_energy_report,
    build_decision_matrix,
)
from .model_comparison import run_model_comparison
from .config import (
    SimConfig,
    TrainConfig,
    QuantumConfig,
    SweepConfig,
    BaselineConfig,
    EnergyConfig,
    ComparisonConfig,
)

__all__ = [
    "WDMChannelSimulator",
    "EdgeLSTM",
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
    "compute_throughput",
    "compute_qpu_economy",
    "compute_energy_report",
    "build_decision_matrix",
    "run_model_comparison",
    "SimConfig",
    "TrainConfig",
    "QuantumConfig",
    "SweepConfig",
    "BaselineConfig",
    "EnergyConfig",
    "ComparisonConfig",
]

__version__ = "3.0.0"
