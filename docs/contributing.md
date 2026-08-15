# Contributing

## Development setup

```bash
git clone https://github.com/example/quantum-twin.git
cd quantum-twin
pip install -e ".[dev]"
```

## Before opening a pull request

CI (`.github/workflows/ci.yml`) runs four gates on every push and PR:
lint, type-check, test, and a package-build sanity check. Run all of them
locally first:

```bash
# 1. Lint + format check
ruff check src tests experiments
ruff format --check src tests experiments

# 2. Type check
mypy src

# 3. Tests (with coverage)
pytest --cov --cov-report=term-missing

# 4. Package builds cleanly
python -m build
```

`ruff format src tests experiments` (without `--check`) auto-fixes
formatting; `ruff check --fix` auto-fixes most lint issues too.

## Code style

- **Type hints are required** on every public function/method signature
  (parameters and return type). `mypy`'s `disallow_untyped_defs` enforces
  this in CI; `tests/` is exempt (see `pyproject.toml`'s
  `[[tool.mypy.overrides]]`).
- **Docstrings** follow the Google style, matching the rest of the
  codebase -- see any existing module for the level of detail expected
  (a docstring here typically explains not just *what* a function does
  but *why* it's designed that way, since this codebase treats docstrings
  as the primary design-rationale record).
- **New modules under `src/quantum_twin/`** are library code: no
  `argparse`, no `if __name__ == "__main__":` orchestration blocks with
  side effects beyond a thin CLI wrapper. That kind of one-off
  orchestration belongs in `experiments/`.
- **New optional dependencies** must degrade gracefully (see
  `quantum_twin.mlops` for the pattern: guarded import, an
  `_AVAILABLE` flag, every public method a safe no-op with a
  once-per-process warning when the dependency is absent) -- never make
  an existing code path suddenly require a new hard dependency.

## Testing conventions

- `tests/` mirrors `src/quantum_twin/`'s structure (`tests/metrics/`
  mirrors `src/quantum_twin/metrics/`).
- Prefer testing pure logic in isolation over requiring a full training
  run or a real Qiskit Aer simulation. See
  `tests/test_quantum_node_mocked.py` for the pattern: patch the
  specific Qiskit collaborators a module imports
  (`unittest.mock.patch.object`) rather than either running a full Aer
  simulation or faking the whole `qiskit` package.
- When mocking multiple collaborators, prefer a `pytest.fixture`
  yielding a `{name: Mock}` dict over stacking several bare `@patch`
  decorators -- stacked decorators inject mocks as positional arguments
  in the REVERSE of their visual order, a well-known footgun that a
  named-dict fixture sidesteps entirely.

## Building the documentation locally

```bash
pip install -e ".[docs]"
mkdocs serve   # live-reloading local preview at http://127.0.0.1:8000
mkdocs build --strict  # what CI runs; fails on any broken link or warning
```
