# Quantum Repeater Digital Twin -- v3 (decomposed + baseline comparison)

Digital twin of a quantum repeater with a predictive admission controller
(`EdgeLSTM` + `CS_MSELoss`) and a Pareto Frontier sweep over the False
Positive penalty, compared against a blind/reactive purification baseline
**and** against four alternative predictor baselines (LSTM+MSE, Random
Forest, XGBoost, Transformer), with throughput, QPU economy, and energy
accounting, ranked by a multi-criteria decision matrix.

## 0. What's new in v3: baseline comparison

`qrepeater_twin/baselines.py` adds four alternative fidelity predictors,
all runnable through the exact same `DigitalTwinOrchestrator` loop as the
`EdgeLSTM + CS_MSELoss` admission controller (so results are directly
comparable, not just architecturally similar):

- **LSTM + MSE** (`train_lstm_mse`) -- the *same* `EdgeLSTM` architecture,
  trained with a plain, cost-insensitive `nn.MSELoss` instead of
  `CS_MSELoss`. Isolates the contribution of the cost-sensitive loss
  itself, holding the architecture fixed.
- **Random Forest** (`train_random_forest`) -- classical, non-recurrent
  ensemble over the flattened window; tests whether explicit temporal
  structure (the LSTM's recurrence) actually earns its keep.
- **XGBoost** (`train_xgboost`) -- gradient-boosted trees over the same
  flattened window. Optional dependency (`pip install xgboost`); skipped
  with a warning, not a crash, if absent (`ComparisonConfig.include_xgboost`).
- **Transformer** (`TinyTransformer` + `train_transformer`) -- a small
  attention-based encoder, as a higher-capacity architectural baseline
  against the compact, edge-oriented `EdgeLSTM`.

`qrepeater_twin/metrics.py` adds the operational metrics needed to
actually choose a predictor for deployment:

- **Throughput** (`compute_throughput`) -- useful entangled pairs
  delivered per second of wall-clock operation (channel cycle time +
  classical inference overhead).
- **QPU economy** (`compute_qpu_economy`) -- purification attempts (and
  QPU shots) avoided relative to the blind baseline.
- **Energy** (`compute_energy_report`) -- an illustrative Joules
  accounting (quantum gate/shot energy + classical inference power) and
  the resulting savings relative to blind purification (`EnergyConfig`).
- **Decision matrix** (`build_decision_matrix`) -- normalizes every
  weighted criterion (direction-aware: latency/energy are cost-type) and
  ranks candidate models by a weighted composite score.

`qrepeater_twin/model_comparison.py` (`run_model_comparison`) trains and
evaluates every predictor above -- plus `EdgeLSTM+CS-MSE` at one
representative `lambda_penalty` and the blind baseline -- over the same
multi-seed protocol as `run_pareto_sweep`, and returns the consolidated
results table and the ranked decision matrix. Run it from the CLI with
`--compare-baselines` (see Usage below), or from Python:

```python
from qrepeater_twin.model_comparison import run_model_comparison

results_df, baseline_metrics, decision_matrix_df, per_model_seed_results = run_model_comparison(
    X_train, y_train, X_test, y_test, device=device,
)
print(decision_matrix_df)  # Rank 1 = recommended model
```

## Fixes carried over from v2

This version also keeps the three fixes applied relative to the original
prototype (single notebook):

## 1. Statistical fragility in the Pareto sweep

**Before:** each `lambda_penalty` value was trained with a single batch and
a single random seed. The original notebook acknowledged, in markdown,
that this left individual points vulnerable to local optima -- but did not
fix the problem.

**Now:** `qrepeater_twin/pareto_sweep.py` trains and evaluates each lambda
value on **multiple independent seeds** (5 by default, configurable via
`SweepConfig.seeds`), and reports **mean ± standard deviation** for every
metric (useful pairs, QPU yield, SKR deficit/surplus, inference latency).
This makes training variance visible directly in the results table, rather
than hidden behind a single point -- essential before using any specific
lambda to decide which admission controller to deploy in production.

## 2. Micro-profiling inaccuracy

**Before:** `time.perf_counter()` timed the EdgeLSTM forward pass, with
`torch.cuda.synchronize()` called before/after on GPU. This guarantees
*correctness* (the measurement won't stop before the CUDA kernel actually
finishes), but the value itself still includes OS scheduler jitter and
Python/CUDA context-switch overhead -- significant when the forward pass
itself only takes a few microseconds/milliseconds.

**Now:** `qrepeater_twin/timing.py` implements `InferenceTimer`, a context
manager that uses `torch.cuda.Event(enable_timing=True)` on GPU (hardware
measurement, directly on the CUDA stream, immune to host-side jitter),
falling back to `time.perf_counter()` only on CPU, where there is no
asynchronous stream to instrument.

## Repository structure

```
│ qrepeater_twin/
│   ├── notebooks/
│   │   └── quantum_repeater_digital_twin_pareto.ipynb  #notebook with explanation
│   ├── src/
│   │   ├──  __init__.py          # public package API
│   │   ├──  config.py             # Sim/Train/Quantum/Sweep/Baseline/Energy/Comparison configs
│   │   ├──  channel_simulator.py  # WDMChannelSimulator (synthetic data generation)
│   │   ├──  models.py              # EdgeLSTM, CS_MSELoss, train_edge_lstm
│   │   ├──  timing.py                # InferenceTimer (CUDA Events / perf_counter)
│   │   ├──  quantum_node.py           # QuantumRepeaterNode (dataplane via Qiskit Aer)
│   │   ├──  orchestrator.py            # DigitalTwinOrchestrator (run_intelligent / run_blind_baseline)
│   │   ├──  pareto_sweep.py              # run_pareto_sweep (multi-seed averaging over lambda_penalty)
│   │   ├──  baselines.py                   # LSTM+MSE, Random Forest, XGBoost, Transformer predictors
│   │   ├──  metrics.py                       # throughput, QPU economy, energy, decision matrix
│   │   ├──  model_comparison.py                # run_model_comparison (cross-architecture, multi-seed)
│   │   ├──  cli.py                               # main() + argparse
│   ├── tests/
│   │   ├──  test_channel_simulator.py
│   │   ├──  test_models_and_timing.py
│   │   ├──  test_baselines.py
│   │   ├──  test_metrics.py
├── CITATION.cff
├── requirements.txt
├── setup.py
└── README.md
```

## Usage

Installation (local environment or Colab):

```bash
pip install -r requirements.txt
pip install -e .
# Optional, only needed for the XGBoost baseline:
pip install -e ".[xgboost]"
```

Run via CLI (Pareto Frontier over `lambda_penalty` only, same as v2):

```bash
python -m qrepeater_twin.cli --epochs 150 --seeds 42 43 44 45 46 \
    --lambda-values 1.0 2.0 5.0 10.0 20.0 50.0
```

Run via CLI with the baseline comparison (LSTM+MSE, Random Forest,
XGBoost, Transformer + throughput/QPU-economy/energy + decision matrix):

```bash
python -m qrepeater_twin.cli --epochs 150 --seeds 42 43 44 45 46 \
    --lambda-values 1.0 2.0 5.0 10.0 20.0 50.0 \
    --compare-baselines --representative-lambda 10.0
```

Run via Python:

```python
from qrepeater_twin.cli import main

results_df, baseline_metrics, per_seed_results, comparison_results = main(
    run_baseline_comparison=True,
)
comp_results_df, comp_baseline_metrics, decision_matrix_df, per_model_seed_results = comparison_results
print(decision_matrix_df)  # Rank 1 = recommended model
```

Tests:

```bash
pytest tests/ -v
```
