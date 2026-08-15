# Installation

Quantum Twin targets Python 3.10+ and uses a `src/` layout with
[Hatchling](https://hatch.pypa.io/) as its build backend (see
`pyproject.toml`).

## From source (editable install, recommended for development)

```bash
git clone https://github.com/example/quantum-twin.git
cd quantum-twin
pip install -e ".[dev]"
```

This installs the core dependencies (`torch`, `numpy`, `pandas`,
`scikit-learn`, `scipy`, `matplotlib`, `qiskit`, `qiskit-aer`) plus the
`dev` extras (`pytest`, `pytest-cov`, `pytest-mock`, `mypy`, `ruff`).

## Optional extras

```bash
# Tree-ensemble baseline (RandomForest is always available via scikit-learn;
# this extra additionally enables the XGBoost baseline).
pip install -e ".[xgboost]"

# MLflow-backed experiment tracking (see quantum_twin.mlops). The project
# works completely fine without this -- experiment_tracking.ExperimentRun
# (plain CSV/PNG/JSON/TeX files) is always available and is the default.
pip install -e ".[mlflow]"

# Documentation toolchain (MkDocs Material + mkdocstrings), only needed to
# build this documentation site locally.
pip install -e ".[docs]"
```

## Verifying the install

```bash
python -c "import quantum_twin; print(quantum_twin.__version__)"
pytest
```

## Console script

Installing the package also puts a `quantum-twin` command on your `PATH`
(see `[project.scripts]` in `pyproject.toml`), equivalent to
`python -m quantum_twin.cli`:

```bash
quantum-twin --epochs 150 --lambda-values 1 2 5 10 20 50
```
