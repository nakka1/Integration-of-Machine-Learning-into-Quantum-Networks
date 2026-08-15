# Quantum Twin

**Digital Twin of a Quantum Repeater with a predictive admission controller.**

Quantum Twin simulates a quantum repeater node operating on a noisy WDM
optical channel, and trains a compact LSTM (`EdgeLSTM`) with a
cost-sensitive loss (`CS_MSELoss`) to predict channel fidelity ahead of
time -- so the repeater can HALT a purification attempt before wasting
QPU time on a photon that was never going to be useful, instead of
purifying unconditionally on every cycle.

## What this project answers

- **Is the predictive controller actually better than doing nothing
  smart?** A 2x2 factorial ablation study isolates the individual and
  combined contribution of the `EdgeLSTM` architecture and the `CS_MSELoss`
  cost function, and every comparison is backed by paired significance
  testing (t-test + Wilcoxon, Holm-Bonferroni corrected) with 95%
  confidence intervals -- not just a mean that looks bigger.
- **Does it hold up over time, or was one train/test split lucky?**
  Walk-forward (rolling-origin) cross-validation re-validates the result
  across several independent, forward-moving slices of the simulated
  channel.
- **How much of the gain is trivial?** Naive/oracle reference baselines
  (`Persistence`, `MovingAverage`, `Oracle`) put every other model's
  result in context: how much better than "repeat the last value", and
  how close to the theoretical best case with perfect information?
- **Is the predictor's timing reliable, not just its average error?**
  Beyond MAE/RMSE/R², a temporal (event-based) analysis checks whether
  the predictor's threshold crossings happen at the right TIME relative
  to the true channel degradation.

## Where to go next

- New to the project? Start with [Installation](getting-started/installation.md)
  and the [Quickstart](getting-started/quickstart.md).
- Want the big picture of how the modules fit together? See
  [Architecture](architecture.md).
- Running a specific experiment? See the [guides](guides/experiments.md).
- Looking for a specific function or class? See the
  [API reference](api/index.md).
- Want to contribute? See [Contributing](contributing.md).
