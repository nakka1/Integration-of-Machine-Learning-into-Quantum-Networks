# Quantum Twin

[![CI](https://github.com/example/quantum-twin/actions/workflows/ci.yml/badge.svg)](https://github.com/example/quantum-twin/actions/workflows/ci.yml)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/)
[![Type checked: mypy](https://img.shields.io/badge/type--checked-mypy-2A6DB2)](https://mypy-lang.org/)
[![Linted: Ruff](https://img.shields.io/badge/linted-ruff-D7FF64)](https://github.com/astral-sh/ruff)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)
[![Docs](https://img.shields.io/badge/docs-mkdocs--material-blue)](https://example.github.io/quantum-twin/)

**Digital Twin of a Quantum Repeater with a predictive admission
controller.** A compact LSTM (`EdgeLSTM`), trained with a cost-sensitive
loss (`CS_MSELoss`), predicts the fidelity of a noisy WDM optical channel
ahead of time, so a repeater node can HALT a purification attempt before
wasting QPU time on a photon that was never going to be useful --
instead of purifying unconditionally on every cycle.

Every claim this project makes about that predictive controller is
backed by: multi-seed statistical robustness, paired significance
testing with multiple-comparisons correction, walk-forward temporal
cross-validation, a 2x2 factorial ablation study isolating architecture
from loss function, and naive/oracle reference baselines putting every
result in context.

## What this solves

Standard practice for these repeater proposals is "here's our model,
here's one number it got." This project instead asks, and answers, the
questions a rigorous evaluation actually needs:

| Question | Where it's answered |
|---|---|
| Is the predictive controller better than blind/unconditional purification? | `orchestrator.DigitalTwinOrchestrator` (`run_intelligent` vs. `run_blind_baseline`) |
| Is a reported improvement real, or seed-to-seed noise? | `statistics_tests.compare_models_statistically` (paired t-test + Wilcoxon, Holm-Bonferroni corrected, 95% CIs) |
| Does it hold up across time, or was one train/test split lucky? | `walk_forward.run_walk_forward_evaluation` (rolling-origin CV) |
| How much of the gain is trivial? | `baselines.PersistencePredictor` / `MovingAveragePredictor` / `OraclePredictor` |
| What does the architecture contribute vs. the loss function? | `ablation.run_ablation_study` (2x2 factorial: `{EdgeLSTM, StandardLSTM} x {MSE, CS-MSE}`) |
| Is the prediction's *timing* reliable, not just its average error? | `metrics.prediction.compute_temporal_prediction_metrics` |

See the [full documentation](https://example.github.io/quantum-twin/) for
the complete picture, or [`docs/architecture.md`](docs/architecture.md)
for the module map.

## Installation

```bash
pip install -e ".[dev]"
```

Requires Python 3.10+. See
[`docs/getting-started/installation.md`](docs/getting-started/installation.md)
for optional extras (`xgboost`, `mlflow`, `docs`).

## Quickstart

```python
from quantum_twin.channel_simulator import WDMChannelSimulator
from quantum_twin.cli import get_device
from quantum_twin.config import QuantumConfig, SimConfig, TrainConfig
from quantum_twin.models import EdgeLSTM, train_edge_lstm
from quantum_twin.orchestrator import DigitalTwinOrchestrator
from quantum_twin.quantum_node import QuantumRepeaterNode

device = get_device()
sim_cfg, train_cfg, quantum_cfg = SimConfig(), TrainConfig(), QuantumConfig()

wdm_sim = WDMChannelSimulator(n_steps=sim_cfg.n_steps, dt=sim_cfg.dt, seed=sim_cfg.seed)
df = wdm_sim.generate_dataset()
X_train, y_train, X_test, y_test, _ = wdm_sim.preprocess(
    df, window_size=sim_cfg.window_size, test_size=sim_cfg.test_size,
)
X_train, y_train = X_train.to(device), y_train.to(device)
X_test, y_test = X_test.to(device), y_test.to(device)

model = EdgeLSTM(input_size=2, hidden_size=train_cfg.hidden_size).to(device)
model = train_edge_lstm(model, X_train, y_train, threshold=train_cfg.threshold,
                         lambda_penalty=10.0, epochs=train_cfg.epochs, device=device, seed=42)

node = QuantumRepeaterNode(T1=quantum_cfg.T1, T2=quantum_cfg.T2, shots=quantum_cfg.shots)
orchestrator = DigitalTwinOrchestrator(model=model, quantum_node=node,
                                         threshold=train_cfg.threshold, device=device)
metrics = orchestrator.run_intelligent(X_test, y_test)
print(f"{metrics['useful_pairs']} useful pairs from {metrics['attempted']} attempts")
```

Or run a full multi-seed experiment end to end:

```bash
python experiments/run_pareto_sweep.py --epochs 150
```

See
[`docs/getting-started/quickstart.md`](docs/getting-started/quickstart.md)
for the complete walkthrough and
[`docs/guides/experiments.md`](docs/guides/experiments.md) for every
experiment type.

## Repository layout

```
quantum-twin/
    pyproject.toml           # build system, dependencies, mypy/ruff/pytest config
    src/quantum_twin/        # the library (import quantum_twin.*) -- see docs/architecture.md
    experiments/              # standalone scripts that CONSUME the library
    tests/                     # pytest suite, mirrors src/quantum_twin/
    docs/                       # MkDocs Material documentation site
    .github/workflows/ci.yml     # lint -> typecheck -> test -> build -> docs
```

Full module-by-module breakdown: [`docs/architecture.md`](docs/architecture.md).

## Development

```bash
pip install -e ".[dev]"
ruff check src tests experiments && ruff format --check src tests experiments
mypy src
pytest --cov
```

All four checks (plus a package-build sanity check and a strict docs
build) run automatically on every push/PR via
[`.github/workflows/ci.yml`](.github/workflows/ci.yml). See
[`docs/contributing.md`](docs/contributing.md) for the full contributor
guide, including this project's testing conventions (mock-based unit
tests for the Qiskit-dependent physics, no full Aer simulation required
for that layer of testing).

## Documentation

Full documentation (architecture, guides, complete API reference
generated from docstrings) is built with MkDocs Material:

```bash
pip install -e ".[docs]"
mkdocs serve
```

## Project status and version history

Currently on **v4.0** (a from-scratch professional-engineering
restructuring: `src/` layout, `pyproject.toml`, full type-hint coverage,
mocked unit tests, optional MLflow tracking, CI, and this documentation
site). See [`docs/changelog.md`](docs/changelog.md) for the complete
version history from the original single-notebook prototype onward.

## License

MIT -- see [`LICENSE`](LICENSE).
