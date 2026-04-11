"""
Tests for the legacy checkpoint key remapping helper.

An older training run saved the Custom CNN with keys like
``conv_block1.0.weight``, but the current architecture uses
``features.0.weight``. ``_remap_legacy_keys`` bridges the two, and this
test suite pins the mapping so a refactor can't silently break loading
the ``custom_cnn_model.pth`` checkpoint that ships with the app.
"""

from __future__ import annotations

import torch

from model_utils import CustomCNN, _remap_legacy_keys


def test_remap_is_noop_when_already_migrated() -> None:
    """A state dict with ``features.*`` keys should pass through unchanged."""
    state = {"features.0.weight": torch.zeros(1)}
    remapped = _remap_legacy_keys(state)
    assert remapped == state


def test_remap_is_noop_when_no_legacy_keys() -> None:
    """Unrelated state dicts should not be rewritten."""
    state = {"classifier.1.weight": torch.zeros(1)}
    remapped = _remap_legacy_keys(state)
    assert remapped == state


def test_remap_translates_block_prefixes() -> None:
    """``conv_blockN.<idx>.<rest>`` → ``features.<offset + idx>.<rest>``.

    Offsets from the implementation:
        conv_block1 → features.0+
        conv_block2 → features.8+
        conv_block3 → features.16+
        conv_block4 → features.24+
    """
    legacy = {
        "conv_block1.0.weight": torch.zeros(1),
        "conv_block1.1.running_mean": torch.zeros(1),
        "conv_block2.3.bias": torch.zeros(1),
        "conv_block3.0.weight": torch.zeros(1),
        "conv_block4.0.weight": torch.zeros(1),
        "classifier.1.weight": torch.zeros(1),  # unchanged
    }
    remapped = _remap_legacy_keys(legacy)

    assert "features.0.weight" in remapped
    assert "features.1.running_mean" in remapped
    assert "features.11.bias" in remapped           # 8 + 3
    assert "features.16.weight" in remapped
    assert "features.24.weight" in remapped
    assert "classifier.1.weight" in remapped
    # The original legacy keys are gone
    assert "conv_block1.0.weight" not in remapped


def test_remapped_state_loads_into_current_custom_cnn() -> None:
    """End-to-end: take a *fresh* CustomCNN, rename its state dict to the
    legacy format, then pass it through ``_remap_legacy_keys`` and confirm
    it loads cleanly back into another CustomCNN.

    This is the "would the production app actually survive a legacy
    checkpoint" test, and it's the one that actually matters for
    shipping.
    """
    model = CustomCNN(num_classes=10)
    real_state = model.state_dict()

    # Build a reverse map: features.0 → conv_block1.0, features.8 → conv_block2.0, ...
    reverse_offsets = [
        ("conv_block1", 0),
        ("conv_block2", 8),
        ("conv_block3", 16),
        ("conv_block4", 24),
    ]
    legacy_state = {}
    for key, tensor in real_state.items():
        if not key.startswith("features."):
            legacy_state[key] = tensor
            continue
        tail = key[len("features."):]
        idx_str, _, rest = tail.partition(".")
        idx = int(idx_str)
        # Walk the offset table in reverse so block4 (offset 24) catches
        # the largest indices first.
        matched = False
        for block_name, offset in reversed(reverse_offsets):
            if idx >= offset:
                legacy_key = f"{block_name}.{idx - offset}"
                if rest:
                    legacy_key = f"{legacy_key}.{rest}"
                legacy_state[legacy_key] = tensor
                matched = True
                break
        if not matched:
            legacy_state[key] = tensor

    # Sanity: the manually-built legacy dict has conv_block* keys
    assert any(k.startswith("conv_block") for k in legacy_state)

    # Run the remapper and load it into a fresh model instance
    migrated = _remap_legacy_keys(legacy_state)
    fresh = CustomCNN(num_classes=10)
    # strict=True verifies every expected key is present and no extras exist
    fresh.load_state_dict(migrated, strict=True)
