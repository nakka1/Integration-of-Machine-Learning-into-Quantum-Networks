"""
Configuration dataclasses.

Centralizing hyperparameters here (instead of scattering them as
positional/keyword arguments across several functions, as in the original
notebook) improves reproducibility: an entire `SweepConfig` can be
serialized (JSON/YAML), version-controlled, and cited in a README or an
experiment report.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List


@dataclass
class SimConfig:
    """Configuration for the synthetic data generator (WDMChannelSimulator)."""

    n_steps: int = 4000
    dt: float = 0.01
    seed: int = 42
    window_size: int = 20
    test_size: float = 0.2


@dataclass
class TrainConfig:
    """Training configuration for EdgeLSTM + CS_MSELoss."""

    hidden_size: int = 16
    epochs: int = 150
    lr: float = 0.012
    threshold: float = 0.65
    lambda_fn: float = 4.0
    discard_penalty_weight: float = 10.0
    max_discard_rate: float = 0.60


@dataclass
class QuantumConfig:
    """Configuration for the virtual quantum dataplane (QuantumRepeaterNode)."""

    T1: float = 50e-6
    T2: float = 30e-6
    depol_prob: float = 0.01
    shots: int = 512
    seed: int = 7
    success_rate_cutoff: float = 0.5


@dataclass
class SweepConfig:
    """
    Configuration for the Pareto Frontier sweep over `lambda_penalty`.

    `seeds` fixes the statistical fragility of the original prototype:
    instead of a single training run (single batch + single seed) per
    lambda value, each point on the Pareto Frontier is the mean (± standard
    deviation) of `len(seeds)` independent training runs, reducing the risk
    of any given point getting stuck in an unrepresentative local optimum.
    """

    lambda_values: List[float] = field(default_factory=lambda: [1.0, 2.0, 5.0, 10.0, 20.0, 50.0])
    seeds: List[int] = field(default_factory=lambda: list(range(42, 52)))  # 10 seeds by default


@dataclass
class BaselineConfig:
    """
    Hyperparameters for the *predictor* baselines compared against the
    intelligent admission controller (EdgeLSTM + CS_MSELoss):

        - LSTM+MSE       : same EdgeLSTM architecture, trained with a plain
                            (cost-insensitive) nn.MSELoss -- isolates the
                            contribution of CS_MSELoss itself, holding the
                            architecture fixed.
        - Random Forest   : classical, non-recurrent regressor over the
                            flattened window (n_estimators/max_depth below).
        - XGBoost         : gradient-boosted trees over the flattened
                            window (n_estimators/max_depth/learning_rate
                            below). Optional dependency -- skipped with a
                            warning (not a hard failure) if `xgboost` is
                            not installed.
        - Transformer     : small Transformer encoder over the same input
                            window, trained with plain nn.MSELoss, as a
                            higher-capacity architectural baseline.
        - Persistence      : F_hat(t) = F(t-1), the simplest possible
                             causal forecast (no training at all).
        - MovingAverage     : F_hat(t) = mean of the last `moving_average_window`
                              true fidelity values (no training at all).
        - Oracle             : F_hat(t) = F_true(t) exactly -- a
                               non-deployable, perfect-information upper
                               bound on achievable QPU yield/throughput,
                               not a candidate for the decision matrix.
    """

    # LSTM + plain MSE (same architecture as EdgeLSTM, no CS_MSELoss)
    lstm_mse_hidden_size: int = 16
    lstm_mse_epochs: int = 150
    lstm_mse_lr: float = 0.012

    # Random Forest
    rf_n_estimators: int = 200
    rf_max_depth: int = 8

    # XGBoost
    xgb_n_estimators: int = 200
    xgb_max_depth: int = 5
    xgb_learning_rate: float = 0.1

    # Transformer encoder
    transformer_d_model: int = 32
    transformer_nhead: int = 4
    transformer_num_layers: int = 2
    transformer_dim_feedforward: int = 64
    transformer_epochs: int = 150
    transformer_lr: float = 0.005

    # Naive/oracle reference baselines (no training; see baselines.py)
    moving_average_window: int = 5


@dataclass
class EnergyConfig:
    """
    Coefficients of the (illustrative) energy-accounting model used by
    `quantum_twin.metrics.compute_energy_report`.

    These are order-of-magnitude, documented estimates -- not vendor
    datasheet values -- meant to make the *relative* energy trade-off
    between "ask the network first" (predictive admission, pays a small,
    constant classical-inference energy on every cycle) and "always
    purify" (blind baseline, pays the full quantum-operation energy on
    every cycle) visible and comparable across predictor models.

        - joules_per_1q_gate / joules_per_2q_gate : energy per logical
          gate operation executed by the QPU (or its control electronics)
          during one BBPSSW purification attempt.
        - joules_per_shot_overhead                : fixed per-shot
          overhead (state prep + measurement + reset) independent of gate
          count.
        - classical_inference_power_w             : average power draw of
          the edge accelerator while a predictor model runs one forward
          pass / one prediction (Watts). Combined with the measured
          per-cycle latency to obtain per-cycle classical energy.
        - classical_idle_power_w                  : power draw of that
          same edge device when it is *not* running a predictor at all
          (the blind baseline never invokes one), included so the blind
          baseline's classical energy isn't silently zero.
    """

    joules_per_1q_gate: float = 5e-9
    joules_per_2q_gate: float = 2e-8
    joules_per_shot_overhead: float = 1e-9
    classical_inference_power_w: float = 0.5
    classical_idle_power_w: float = 0.05

    # BBPSSW circuit gate counts (see quantum_node.build_bbpssw_circuit):
    # 2x H, 2x CX (Bell-pair prep) + 4x id + 2x CX (bilateral CNOTs).
    gates_1q_per_attempt: int = 6
    gates_2q_per_attempt: int = 4


@dataclass
class AblationConfig:
    """
    Configuration for `quantum_twin.ablation.run_ablation_study`, which
    trains and evaluates the 2x2 factorial grid
    `{StandardLSTM, EdgeLSTM} x {MSE, CS_MSELoss}` under the same
    multi-seed protocol as `run_pareto_sweep` / `run_model_comparison`, to
    isolate the individual and combined contribution of the architecture
    (compact/edge vs. larger/conventional LSTM) and the loss function
    (cost-insensitive MSE vs. cost-sensitive CS-MSE).

    `EdgeLSTM`'s own hyperparameters are taken from `TrainConfig`
    (`hidden_size`, `threshold`, `lambda_fn`, etc.); this dataclass only
    adds what's specific to the ablation grid: `StandardLSTM`'s capacity,
    and the shared training budget (`epochs`/`lr`) applied to all four
    grid cells so no cell is unfairly under/over-trained relative to the
    others.
    """

    standard_lstm_hidden_size: int = 64
    standard_lstm_num_layers: int = 2
    standard_lstm_dropout: float = 0.1

    epochs: int = 150
    lr: float = 0.012
    # lambda_penalty used for both CS-MSE grid cells (EdgeLSTM+CS-MSE and
    # StandardLSTM+CS-MSE); threshold/lambda_fn/discard-rate stabilizers
    # come from TrainConfig, shared with the rest of the pipeline.
    representative_lambda: float = 10.0

    seeds: List[int] = field(default_factory=lambda: list(range(42, 52)))  # 10 seeds by default
    cycle_time_s: float = 1e-3

    # Headline metrics on which the 2x2 factorial decomposition
    # (architecture effect / loss effect / interaction effect) is
    # reported. Each must be a numeric column produced by
    # `ablation.run_ablation_study`'s per-seed rows.
    headline_metrics: List[str] = field(default_factory=lambda: ["qpu_yield_pct", "mae", "fp"])


@dataclass
class WalkForwardConfig:
    """
    Configuration for walk-forward (rolling-origin) temporal
    cross-validation (`walk_forward.run_walk_forward_evaluation`),
    replacing the single chronological train/test split used elsewhere in
    this project with several NON-OVERLAPPING folds that each slide
    forward in time -- the standard technique for validating a
    time-series model without a single lucky (or unlucky) split
    dominating the reported result.

    Fold k's test segment is a contiguous block of `test_size` samples
    starting `gap` samples after fold k's training segment ends; folds
    move strictly forward through the data (fold k's test segment is
    always chronologically later than fold k-1's), so no fold's training
    segment ever contains data from its own or a later fold's future.

        - expanding=True  : each fold's training segment GROWS (always
          starts at sample 0) -- uses all data available up to that point
          in time, mirroring how a real deployed system accumulates
          history.
        - expanding=False : each fold's training segment is a FIXED-SIZE
          rolling window of the most recent `min_train_size` samples --
          tests whether the model still works when trained on only
          recent, non-accumulating data (e.g. under a deployment
          constraint that discards old telemetry).
        - gap             : samples skipped between a fold's training and
          test segments (a "purge gap"), avoiding any subtle leakage from
          windowed features that straddle the train/test boundary.
    """

    n_splits: int = 5
    test_size: int = 150       # windowed samples per test fold
    min_train_size: int = 300  # size of the fold-0 training segment
    expanding: bool = True
    gap: int = 0

    epochs: int = 150
    lr: float = 0.012
    representative_lambda: float = 10.0
    # Base seed for weight initialization; each fold additionally offsets
    # this by its own fold index (seed + fold_idx) so folds don't share an
    # identical initialization while remaining fully reproducible.
    seed: int = 42


@dataclass
class ComparisonConfig:
    """
    Configuration for `quantum_twin.model_comparison.run_model_comparison`,
    which trains/evaluates every predictor baseline (EdgeLSTM+CS_MSELoss at
    a representative lambda, LSTM+MSE, Random Forest, XGBoost, Transformer)
    under the same multi-seed protocol as `run_pareto_sweep`, and reports
    regression accuracy (MAE/RMSE/R^2), the admission confusion matrix
    (FP/FN/TP/TN), throughput, QPU economy, energy, the dimensionless
    latency ratio C_latencia = tau_inf / T2, and a multi-criteria decision
    matrix.
    """

    representative_lambda: float = 10.0
    seeds: List[int] = field(default_factory=lambda: list(range(42, 52)))  # 10 seeds by default
    include_xgboost: bool = True
    cycle_time_s: float = 1e-3
    # +/- fractional perturbation applied to each decision-matrix weight in
    # `quantum_twin.sensitivity.run_weight_sensitivity_analysis` (10% by
    # default, per the requested robustness check).
    sensitivity_perturbation_pct: float = 0.10
    # Weights for the decision matrix (must be non-negative; renormalized
    # internally). Each criterion is oriented so that a HIGHER weighted
    # score is always better (cost-type criteria -- energy and the
    # dimensionless latency ratio C_latencia = tau_inf / T2 -- are inverted
    # before weighting; see metrics._COST_CRITERIA). C_latencia replaces a
    # raw millisecond comparison with a dimensionless ratio against the
    # qubit coherence time T2, turning "inference latency" from an IT
    # metric into a physical quantum constraint: C_latencia << 1 means the
    # forward pass is negligible next to decoherence; C_latencia >~ 1 means
    # the memory has already substantially decohered before a decision is
    # even made.
    decision_weights: dict = field(default_factory=lambda: {
        "qpu_yield_pct": 0.25,
        "throughput_pairs_per_s": 0.20,
        "qpu_cycles_saved_pct": 0.20,
        "energy_saved_pct": 0.20,
        "latency_ratio_c": 0.15,
    })
