import os
import random

import numpy as np
import torch

from quantum_twin.reproducibility import seed_everything, set_full_determinism


def test_seed_everything_makes_python_random_reproducible():
    seed_everything(123)
    a = random.random()
    seed_everything(123)
    b = random.random()
    assert a == b


def test_seed_everything_makes_numpy_reproducible():
    seed_everything(123)
    a = np.random.rand(5)
    seed_everything(123)
    b = np.random.rand(5)
    assert np.allclose(a, b)


def test_seed_everything_makes_torch_reproducible():
    seed_everything(123)
    a = torch.rand(5)
    seed_everything(123)
    b = torch.rand(5)
    assert torch.allclose(a, b)


def test_seed_everything_makes_model_init_reproducible():
    import torch.nn as nn

    seed_everything(42)
    model_a = nn.Linear(4, 1)
    seed_everything(42)
    model_b = nn.Linear(4, 1)
    assert torch.allclose(model_a.weight, model_b.weight)
    assert torch.allclose(model_a.bias, model_b.bias)


def test_seed_everything_different_seeds_differ():
    seed_everything(1)
    a = torch.rand(10)
    seed_everything(2)
    b = torch.rand(10)
    assert not torch.allclose(a, b)


def test_set_full_determinism_sets_cudnn_flags():
    set_full_determinism(seed=7, deterministic_algorithms=True)
    assert torch.backends.cudnn.deterministic is True
    assert torch.backends.cudnn.benchmark is False


def test_set_full_determinism_sets_cublas_workspace_env_var():
    os.environ.pop("CUBLAS_WORKSPACE_CONFIG", None)
    set_full_determinism(seed=7, deterministic_algorithms=True)
    assert os.environ.get("CUBLAS_WORKSPACE_CONFIG") is not None


def test_set_full_determinism_without_flag_skips_cudnn_changes():
    # Reset to a known non-deterministic-ish baseline first.
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True
    set_full_determinism(seed=7, deterministic_algorithms=False)
    # Should NOT have forced cudnn.deterministic=True this time.
    assert torch.backends.cudnn.deterministic is False


def test_set_full_determinism_also_seeds_everything_when_seed_given():
    set_full_determinism(seed=99)
    a = torch.rand(5)
    set_full_determinism(seed=99)
    b = torch.rand(5)
    assert torch.allclose(a, b)


def test_set_full_determinism_none_seed_does_not_raise():
    # deterministic_algorithms flags can be set without also reseeding.
    set_full_determinism(seed=None, deterministic_algorithms=True)
