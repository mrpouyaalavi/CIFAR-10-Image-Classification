"""
Shared pytest fixtures for the CIFAR-10 portfolio test suite.

Design notes
------------
* Every fixture here is cheap to construct so the whole test matrix can
  run on a free-tier GitHub Actions runner in under a minute. We avoid
  touching the real checkpoints on disk — the goal is to exercise the
  architectures, preprocessing pipelines, and module-level helpers, not
  to re-validate the published accuracy numbers.
* The ``device`` fixture is always CPU. GitHub Actions runners don't have
  CUDA or Apple MPS, and forcing CPU keeps the tests deterministic and
  avoids platform-specific skips.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

# Make the project root importable without installing the package. This is
# what lets `import model_utils` / `import benchmark_data` work from inside
# the ``tests/`` package regardless of how pytest is invoked.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


@pytest.fixture(scope="session")
def device() -> torch.device:
    """Always-CPU torch device for deterministic, portable tests."""
    return torch.device("cpu")


@pytest.fixture()
def dummy_pil_image() -> Image.Image:
    """Return a deterministic 64x64 RGB test image.

    We seed NumPy locally so the fixture is reproducible without leaking
    state into unrelated tests that also use the global RNG.
    """
    rng = np.random.default_rng(seed=1234)
    arr = rng.integers(low=0, high=255, size=(64, 64, 3), dtype=np.uint8)
    return Image.fromarray(arr, mode="RGB")


@pytest.fixture()
def dummy_small_pil_image() -> Image.Image:
    """32x32 RGB image — native CIFAR-10 resolution."""
    rng = np.random.default_rng(seed=5678)
    arr = rng.integers(low=0, high=255, size=(32, 32, 3), dtype=np.uint8)
    return Image.fromarray(arr, mode="RGB")
