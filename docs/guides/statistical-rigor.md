# Statistical rigor

Every multi-seed table in this project reports mean +/- standard
deviation, but that alone doesn't say whether a difference is
distinguishable from seed-to-seed noise. `quantum_twin.statistics_tests`
adds the inferential layer.

## Comparing EdgeLSTM+CS-MSE against every baseline

```python
from quantum_twin.statistics_tests import compare_models_statistically

significance_df = compare_models_statistically(
    per_model_seed_results,       # from run_model_comparison
    metric_key="qpu_yield_pct",
    reference_model="EdgeLSTM+CS-MSE",
    alpha=0.05,
)
```

For each other model, this reports:

- the mean difference (comparison model - reference model),
- a 95% confidence interval on that difference (t-distribution based),
- a paired t-test p-value and a Wilcoxon signed-rank p-value,
- both p-values **after** Holm-Bonferroni correction across the whole
  family of comparisons, with a boolean "significant at alpha" flag.

Models evaluated on a single nominal seed (the deterministic
`Persistence`/`MovingAverage`/`Oracle` baselines) are correctly reported
with `N Paired Seeds == 1` and `NaN` p-values (below
`compare_models_statistically`'s `min_n_for_tests` default) rather than a
statistically meaningless single-point test.

Visualize the whole family at once with a forest plot:

```python
from quantum_twin import plotting

fig = plotting.plot_significance_forest(significance_df)
```

## Does the result hold up across time?

See [Running experiments](experiments.md) for
`walk_forward.run_walk_forward_evaluation`, which re-validates a result
across several independent, non-overlapping, forward-moving slices of
the simulated channel and reports a 95% CI per metric across folds.

## Full determinism

`quantum_twin.reproducibility.seed_everything(seed)` is called at the
start of every training round (Python `random` + NumPy + Torch CPU/CUDA
generators). `set_full_determinism(seed)`, called once at the top of
`cli.main`, additionally forces PyTorch's deterministic CUDA/cuDNN
algorithm variants -- see that function's docstring for the exact flags
and their trade-offs.
