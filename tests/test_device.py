"""
Device selection helper tests.

``select_device()`` is the single source of truth for CUDA/MPS/CPU
auto-detection across the whole repo. These tests verify its contract
without depending on the host actually having a GPU.
"""

from __future__ import annotations

import torch

from model_utils import describe_device, select_device


def test_select_device_returns_valid_torch_device() -> None:
    """The returned value is always a real ``torch.device`` instance.

    We call with ``verbose=False`` so the test does not print to stdout
    (which pytest captures anyway, but keeping tests quiet is nicer).
    """
    dev = select_device(verbose=False)
    assert isinstance(dev, torch.device)
    assert dev.type in {"cuda", "mps", "cpu"}


def test_select_device_is_idempotent() -> None:
    """Repeated calls return equivalent devices.

    The function has a module-level ``_device_logged`` flag it uses to
    suppress duplicate prints, but the returned ``torch.device`` must be
    consistent across calls in the same process.
    """
    first = select_device(verbose=False)
    second = select_device(verbose=False)
    assert first.type == second.type


def test_describe_device_returns_non_empty_string() -> None:
    """The human-readable device description is used in the sidebar."""
    dev = select_device(verbose=False)
    label = describe_device(dev)
    assert isinstance(label, str)
    assert len(label) > 0
    # The label should mention the device family so the sidebar never
    # renders a generic "cpu" — that would be a regression of the honest
    # "CPU - Apple M-series" display contract.
    assert any(token in label for token in ("CUDA", "MPS", "CPU"))


def test_describe_device_handles_cpu_directly() -> None:
    """Can be called with an explicit CPU device without touching CUDA APIs."""
    label = describe_device(torch.device("cpu"))
    assert "CPU" in label
