"""
Component 13 -- Full reproducibility / determinism.

Every training routine in this project already calls `torch.manual_seed(seed)`
before a model's weights are initialized, which is enough to make CPU
runs bit-exact. It is NOT enough on GPU: several `cuDNN`/CUDA kernels
(including the ones `nn.LSTM` uses internally) pick nondeterministic,
faster algorithms by default, so two runs with the identical seed can
still diverge in their low-order bits once a GPU is involved -- silently
undermining the very reproducibility the seed was meant to guarantee.

This module centralizes the two determinism concerns that matter here:

    - `seed_everything(seed)` : the per-round convenience used everywhere
      a single seed needs to be applied (Python's `random`, NumPy, and
      Torch's CPU **and** CUDA generators) before one model is built and
      trained.
    - `set_full_determinism(seed)` : called ONCE per process (see
      `cli.main`), additionally forcing PyTorch's CUDA/cuDNN backend into
      its deterministic algorithm variants. This is a process-wide
      setting -- it cannot meaningfully be toggled per training round --
      and has a real (usually modest, occasionally significant) run-time
      cost, which is why it is opt-in via a single call at the top of a
      run rather than baked into every `seed_everything` call.
"""

from __future__ import annotations

import os
import random

import numpy as np
import torch


def seed_everything(seed: int) -> None:
    """
    Seeds every source of randomness this project's training routines
    touch for ONE model-building/training round: Python's `random`,
    NumPy, and Torch's CPU generator plus (if available) every CUDA
    device's generator via `torch.cuda.manual_seed_all`.

    This is what `train_edge_lstm` / `baselines.train_lstm_mse` /
    `baselines.train_transformer` and every `seed`-looping caller
    (`pareto_sweep.run_pareto_sweep`, `model_comparison.run_model_comparison`,
    `ablation.run_ablation_study`, `walk_forward.run_walk_forward_evaluation`)
    call in place of a bare `torch.manual_seed(seed)`, so a given `seed`
    produces the same weight initialization on CPU AND GPU alike.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def set_full_determinism(seed: int | None = None, deterministic_algorithms: bool = True) -> None:
    """
    Process-wide determinism switch. Call this ONCE, near the start of a
    run (see `cli.main`), before any model is built.

    Beyond `seed_everything` (which only fixes the *initial state* of
    every RNG), this additionally:

        - Sets `CUBLAS_WORKSPACE_CONFIG`, required by CUDA >= 10.2 for
          deterministic `cuBLAS` behavior (must be set before the first
          CUDA context is created -- this function should therefore be
          called before any `.to("cuda")`/tensor-on-GPU operation).
        - Disables `cudnn.benchmark` (which otherwise lets cuDNN
          auto-tune and cache the fastest convolution/RNN algorithm for
          the current input shapes -- a process that is itself
          nondeterministic across runs) and enables `cudnn.deterministic`.
        - Calls `torch.use_deterministic_algorithms(True, warn_only=True)`,
          which makes PyTorch prefer deterministic kernel implementations
          throughout (including inside `nn.LSTM`) and only WARNS (instead
          of raising `RuntimeError`) for the handful of operations that
          have no deterministic GPU implementation at all -- `warn_only`
          is used deliberately here so a missing deterministic kernel
          degrades a run's exactness rather than crashing it outright,
          appropriate for a research pipeline that still needs to
          complete on whatever hardware it happens to run on.

    Caveat (documented, not hidden): even with every flag above set,
    bit-exact reproducibility across DIFFERENT GPU models, CUDA/cuDNN
    versions, or PyTorch builds is not guaranteed -- only across repeated
    runs on the SAME hardware/software stack. `seed_everything`'s effect
    (identical weight initialization and training trajectory shape) is
    what actually travels across machines; full bit-exactness does not,
    and no library can promise otherwise.
    """
    if seed is not None:
        seed_everything(seed)

    if deterministic_algorithms:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
        except TypeError:
            # Older torch versions do not accept `warn_only`.
            torch.use_deterministic_algorithms(True)
