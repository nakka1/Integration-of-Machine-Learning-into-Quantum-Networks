# Quantum Repeater Digital Twin -- v3 (decomposed + baseline comparison)

Digital twin of a quantum repeater with a predictive admission controller
(`EdgeLSTM` + `CS_MSELoss`) and a Pareto Frontier sweep over the False
Positive penalty, compared against a blind/reactive purification baseline
**and** against four alternative predictor baselines (LSTM+MSE, Random
Forest, XGBoost, Transformer), with throughput, QPU economy, and energy
accounting, ranked by a multi-criteria decision matrix.

## Repository structure

```
│ qrepeater_twin/
│   ├── notebooks/
│   │   └── quantum_repeater_digital_twin_pareto.ipynb  #notebook with explanation
│   ├── src/
│   │   ├──  __init__.py          # public package API
│   │   ├──  config.py             # Sim/Train/Quantum/Sweep/Baseline/Energy/Comparison/Ablation configs
│   │   ├──  channel_simulator.py  # WDMChannelSimulator (synthetic data generation)
│   │   ├──  models.py              # EdgeLSTM, StandardLSTM, CS_MSELoss, train_edge_lstm
│   │   ├──  timing.py                # InferenceTimer (CUDA Events / perf_counter)
│   │   ├──  quantum_node.py           # QuantumRepeaterNode (dataplane via Qiskit Aer)
│   │   ├──  orchestrator.py            # DigitalTwinOrchestrator (run_intelligent / run_blind_baseline, confusion matrix)
│   │   ├──  pareto_sweep.py              # run_pareto_sweep (multi-seed, MAE/RMSE/R^2, FP/FN, C_latencia)
│   │   ├──  baselines.py                   # LSTM+MSE, Random Forest, XGBoost, Transformer predictors
│   │   ├──  model_comparison.py                    # run_model_comparison (cross-architecture, multi-seed)
│   │   ├──  ablation.py                              # run_ablation_study (2x2 factorial: architecture x loss)
│   │   ├──  sensitivity.py                             # run_weight_sensitivity_analysis (+/-10% weight robustness)
│   │   ├──  plotting.py                                  # matplotlib chart generators for every result table
│   │   ├──  experiment_tracking.py                         # ExperimentRun + per-experiment-type trackers
│   │   ├──  cli.py                                           # main() + argparse
│   ├──  metrics/                         # split into 4 single-responsibility submodules:
│   │   ├──  __init__.py                   # public package API
│   │   ├──  prediction.py                  #   MAE/RMSE/R^2 + temporal threshold-crossing timing analysis
│   │   ├──  quantum.py                      #   QPU economy, fidelity trajectory statistics
│   │   ├──  performance.py                    #   throughput, C_latencia, energy accounting
│   │   ├──  decision.py                        #   confusion-matrix classification metrics, decision matrix
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

## License

This project is distributed under the Apache 2.0 license.

## Version History

| Versão | Data | Descrição |
| ------- | ------------ | ------------------------------------------ |
| v1 | - | Notebook monolítico original: simulador de canal, EdgeLSTM, CS_MSELoss, dataplane quântico e orquestrador em um único Jupyter Notebook, sem testes, com treino em batch único / seed única e profiling via `time.perf_counter()`. |
| v2 | 2026-08-07 | Decomposição em pacote `qrepeater_twin/` (um módulo por responsabilidade); correção da fragilidade estatística via multi-seed averaging na Pareto Frontier (mean ± std); micro-profiling corrigido com `torch.cuda.Event` (`InferenceTimer`); suíte de testes `pytest`. |
| v3 | 2026-08-07 | Comparação cross-architecture contra baselines (LSTM+MSE, Random Forest, XGBoost, Transformer); matriz de confusão de admissão (TP/FP/TN/FN); razão adimensional de latência `C_latência = τ_inf/T2`; throughput; economia de QPU; contabilidade de energia; matriz de decisão multicritério ponderada. |
| v3.1 | 2026-08-08 | Análise de sensibilidade dos pesos da matriz de decisão (perturbação automática ±10%, taxa de vitória por modelo); ajustes estruturais nas métricas derivadas. |
| v3.2 | 2026-08-09 | Avaliação preditiva completa da EdgeLSTM (MAE/RMSE/R² + análise temporal de cruzamento de limiar: atraso/antecipação, eventos perdidos, falsos alarmes); estudo de ablação fatorial 2×2 (`{EdgeLSTM, StandardLSTM} × {MSE, CS-MSE}` com decomposição de efeitos); pipeline automática de rastreamento de experimentos (`ExperimentRun`, config/tabelas/figuras/manifest); `metrics.py` reestruturado no pacote `metrics/` (`prediction.py`, `quantum.py`, `performance.py`, `decision.py`); módulo `plotting.py` com gráficos para cada tabela de resultado. |

### 0.2 What's new in v3.2: predictive evaluation, ablation, experiment pipeline

- **Full evaluation of the EdgeLSTM's predictive capability.**
  `metrics.prediction` now reports, beyond MAE/RMSE/R^2, a full TEMPORAL
  analysis of threshold-crossing timing: `find_threshold_crossings`
  locates degradation events (with sub-sample linear interpolation),
  `match_crossings` pairs true/predicted events, and
  `compute_temporal_prediction_metrics` /
  `compute_controller_decision_timing` report the signed timing error
  (positive = late/lagging detection, negative = early/anticipatory) and
  value-space error at the true crossing instant -- answering not just
  "is the final decision good?" but "is it based on a temporally reliable
  prediction?". `metrics.quantum.compute_fidelity_statistics` adds
  descriptive statistics of the true fidelity trajectory itself
  (degradation/recovery event counts).
- **2x2 factorial ablation study.** `ablation.run_ablation_study` trains
  and evaluates the full `{EdgeLSTM, StandardLSTM} x {MSE, CS-MSE}` grid
  (`StandardLSTM`, in `models.py`, is a larger, non-edge-optimized LSTM
  counterpart to `EdgeLSTM`) under the same multi-seed protocol as the
  rest of the pipeline, and reports the standard architecture/loss/
  interaction effect decomposition -- directly answering "what is the
  impact of the EdgeLSTM architecture?", "what is the impact of CS-MSE?",
  and "does the gain come from their combination?".
- **Automatic experiment-tracking pipeline.** `experiment_tracking.ExperimentRun`
  (plus `track_pareto_sweep_experiment` / `track_model_comparison_experiment`
  / `track_ablation_experiment`) writes every run's configuration
  (model/hyperparameters/epochs/seeds/simulator/channel parameters),
  result tables (CSV), figures (PNG), and a `manifest.json` to a
  timestamped directory -- so every result in this project is
  reproducible from disk alone, without re-running anything.
- **Metrics module restructured.** `metrics.py` is now the `metrics/`
  package, split into `prediction.py` (MAE/RMSE/R^2 + temporal timing),
  `quantum.py` (QPU economy, fidelity statistics), `performance.py`
  (throughput, C_latencia, energy), and `decision.py` (confusion-matrix
  classification metrics + the multi-criteria decision matrix) --
  `from qrepeater_twin.metrics import ...` keeps working unmodified
  (the split only reorganizes internals, not the public API).
  `metrics.decision.classify_controller_bias` turns the confusion-matrix
  rates into a plain-language verdict: is the controller accepting
  degraded states, rejecting still-usable ones, or excessively
  conservative?
- **Plotting helpers.** `plotting.py` adds a matplotlib chart generator
  for every result table in this project (Pareto frontier, model/ablation
  comparison bars, confusion matrix, decision matrix, sensitivity summary,
  temporal-error histogram, fidelity timeline with crossings, ablation
  interaction plot) -- each returns a `Figure` and optionally saves to
  disk, so notebook cells and the experiment-tracking pipeline share the
  exact same plotting code.

Digital twin of a quantum repeater with a predictive admission controller
(`EdgeLSTM` + `CS_MSELoss`) and a Pareto Frontier sweep over the False
Positive penalty, compared against a blind/reactive purification baseline
**and** against four alternative predictor baselines (LSTM+MSE, Random
Forest, XGBoost, Transformer). Every predictor is scored on pure ML
regression accuracy (MAE/RMSE/R^2), an admission confusion matrix
(FP/FN/TP/TN), throughput, QPU economy, energy, and the dimensionless
latency ratio C_latencia = tau_inf/T2 -- ranked by a multi-criteria
decision matrix whose robustness is certified by an automatic +/-10%
weight-sensitivity analysis.

### 0.1 What's new in v3.1: structural updates

- **Pure ML metrics (MAE / RMSE / R^2).** `metrics.evaluate_predictor_regression`
  runs one full-batch forward pass of the trained predictor over the test
  set and reports its regression quality against the true fidelity
  trajectory -- entirely BEFORE and decoupled from any admission-control
  logic or the quantum dataplane. Reported per model, per lambda (Pareto
  sweep) and per architecture (baseline comparison).
- **Admission control confusion matrix.** `DigitalTwinOrchestrator.run_intelligent`
  (and `run_blind_baseline`, degenerate by construction) now explicitly
  tallies **True/False Positives/Negatives** on every cycle: a **False
  Positive** is a DEAD photon admitted (`F_true < threshold <= F_pred`,
  wasting a purification attempt), a **False Negative** is a GOOD photon
  discarded (`F_pred < threshold <= F_true`, lost throughput). This is the
  direct, countable justification for `CS_MSELoss`'s asymmetric penalty
  (`lambda_penalty` on FP >> `lambda_fn` on FN). `metrics.compute_confusion_metrics`
  derives precision/recall/specificity/FPR/FNR/F1 from it.
- **Dimensionless temporal-scale ratio.** `metrics.compute_latency_ratio`
  replaces a raw-millisecond latency comparison with `C_latencia = tau_inf
  / T2`: the classical inference latency expressed as a FRACTION of the
  qubit's own coherence time, turning "inference latency" from an IT
  metric into a physical quantum constraint. This is now the (cost-type)
  latency criterion in the decision matrix, in place of raw
  milliseconds.
- **Decision-weight sensitivity analysis.** `sensitivity.run_weight_sensitivity_analysis`
  perturbs every decision-matrix weight by +/-10% (all `2**n` corner
  combinations -- exhaustive, not a random sample; see the module
  docstring for the linear-programming argument for why vertex
  enumeration is sufficient) and reports how often each model still ranks
  #1. `run_model_comparison` runs this automatically and prints a
  one-line verdict (e.g. `'EdgeLSTM+CS-MSE' wins Rank #1 in 100% of the
  vertex trials`).

### 0. What's new in v3: baseline comparison

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

### Fixes carried over from v2

This version also keeps the three fixes applied relative to the original
prototype (single notebook):

### 1. Statistical fragility in the Pareto sweep

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

### 2. Micro-profiling inaccuracy

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