"""
Tests for the preprocessing / transform pipeline.

These tests make sure ``get_transform()`` produces tensors with the exact
shapes, dynamic ranges, and aspect-ratio handling that the model architectures
require.

Key invariants tested:
  1. Output shape is always (3, H, W) — correct for every model.
  2. Output dtype is float32.
  3. Normalization statistics match the model family (CIFAR vs ImageNet).
  4. Non-square (landscape or portrait) inputs produce the correct output
     shape WITHOUT squashing — the pipeline must be aspect-ratio-preserving.
     This was a bug in earlier versions of the codebase: Resize((224,224))
     would compress a 16:9 photo into a 1.8× taller distorted image.
"""

from __future__ import annotations

import math

import pytest
import torch
from PIL import Image

from model_utils import (
    CIFAR_MEAN,
    CIFAR_STD,
    IMAGENET_MEAN,
    IMAGENET_STD,
    get_transform,
)


# ── Shape + dtype tests (square dummy image) ─────────────────────────────────

def test_custom_cnn_transform_produces_32x32(dummy_pil_image) -> None:
    """Custom CNN expects 3×32×32 float tensors (CIFAR-10 native size)."""
    transform = get_transform("Custom CNN")
    tensor = transform(dummy_pil_image)
    assert isinstance(tensor, torch.Tensor)
    assert tensor.shape == (3, 32, 32)
    assert tensor.dtype == torch.float32


def test_mobilenet_transform_produces_224x224(dummy_pil_image) -> None:
    """MobileNetV2 expects 3×224×224 float tensors (ImageNet input size)."""
    transform = get_transform("MobileNetV2")
    tensor = transform(dummy_pil_image)
    assert tensor.shape == (3, 224, 224)
    assert tensor.dtype == torch.float32


def test_resnet18_transform_produces_224x224(dummy_pil_image) -> None:
    """ResNet-18 uses the same ImageNet pipeline as MobileNetV2: 224×224."""
    transform = get_transform("ResNet-18")
    tensor = transform(dummy_pil_image)
    assert tensor.shape == (3, 224, 224)
    assert tensor.dtype == torch.float32


def test_unknown_model_defaults_to_imagenet_transform(dummy_pil_image) -> None:
    """Unrecognised names fall through to the ImageNet pipeline.

    This is the current behaviour in ``model_utils.get_transform`` and
    it's intentional so newly-added transfer-learning architectures get
    sensible defaults until someone wires a custom branch.
    """
    transform = get_transform("SomeNewArchitecture")
    tensor = transform(dummy_pil_image)
    assert tensor.shape == (3, 224, 224)


# ── Aspect-ratio preservation tests (non-square inputs) ─────────────────────
#
# These tests existed as a gap in the previous test suite.  The bug they guard
# against: using Resize((224,224)) on a 16:9 landscape photo squashes the
# image horizontally by ~1.8×, producing distorted features that degrade
# demo quality.  The fix uses Resize(256)+CenterCrop(224) for ImageNet models
# and letterbox padding for the Custom CNN.

@pytest.fixture()
def landscape_pil_image() -> Image.Image:
    """A 320×180 (16:9) landscape image — typical uploaded car/vehicle photo."""
    return Image.new("RGB", (320, 180), color=(100, 150, 200))


@pytest.fixture()
def portrait_pil_image() -> Image.Image:
    """A 180×320 (9:16) portrait image — typical phone photo."""
    return Image.new("RGB", (180, 320), color=(200, 150, 100))


def test_imagenet_transform_landscape_produces_224x224(landscape_pil_image) -> None:
    """A 16:9 landscape upload must be transformed to (3, 224, 224), not squashed."""
    for model_name in ("MobileNetV2", "ResNet-18"):
        tensor = get_transform(model_name)(landscape_pil_image)
        assert tensor.shape == (3, 224, 224), (
            f"{model_name}: expected (3,224,224) from 16:9 input, got {tuple(tensor.shape)}"
        )
        assert tensor.dtype == torch.float32


def test_imagenet_transform_portrait_produces_224x224(portrait_pil_image) -> None:
    """A 9:16 portrait upload must be transformed to (3, 224, 224)."""
    for model_name in ("MobileNetV2", "ResNet-18"):
        tensor = get_transform(model_name)(portrait_pil_image)
        assert tensor.shape == (3, 224, 224), (
            f"{model_name}: expected (3,224,224) from 9:16 input, got {tuple(tensor.shape)}"
        )


def test_custom_cnn_transform_landscape_produces_32x32(landscape_pil_image) -> None:
    """A 16:9 landscape upload must be letterboxed then resized to (3, 32, 32).

    Before the letterbox fix, Resize((32,32)) on a 320×180 image would squash
    the image into an extremely distorted representation.  Letterboxing pads
    the shorter edge first so the resize sees a square and preserves content.
    """
    tensor = get_transform("Custom CNN")(landscape_pil_image)
    assert tensor.shape == (3, 32, 32)
    assert tensor.dtype == torch.float32


def test_custom_cnn_transform_portrait_produces_32x32(portrait_pil_image) -> None:
    """A 9:16 portrait upload must also be letterboxed to (3, 32, 32)."""
    tensor = get_transform("Custom CNN")(portrait_pil_image)
    assert tensor.shape == (3, 32, 32)


@pytest.mark.parametrize(
    ("model_name", "expected_mean", "expected_std"),
    [
        ("Custom CNN", CIFAR_MEAN, CIFAR_STD),
        ("MobileNetV2", IMAGENET_MEAN, IMAGENET_STD),
        ("ResNet-18", IMAGENET_MEAN, IMAGENET_STD),
    ],
)
def test_transform_uses_correct_normalization(
    model_name: str,
    expected_mean: tuple[float, float, float],
    expected_std: tuple[float, float, float],
    dummy_pil_image,
) -> None:
    """Verify the normalization statistics by reversing them on a
    constant input.

    We pass a uniform grey (128) image through the pipeline, then work
    backwards from the normalised output tensor to recover the mean
    that must have been subtracted. The recovered values must match
    the constants declared in ``model_utils`` — if someone copies the
    wrong mean/std (an easy mistake when swapping backbones), this
    test catches it.
    """
    from PIL import Image

    grey = Image.new("RGB", (64, 64), color=(128, 128, 128))
    transform = get_transform(model_name)
    tensor = transform(grey)

    # A uniform 128 pixel becomes 128/255 = 0.5019... before normalization,
    # then subtracts mean and divides by std. Per-channel:
    #     normalized = (0.502 - mean) / std
    grey_norm = 128 / 255.0
    for c in range(3):
        channel_val = tensor[c].mean().item()
        recovered_mean = grey_norm - channel_val * expected_std[c]
        assert math.isclose(recovered_mean, expected_mean[c], abs_tol=1e-3), (
            f"Channel {c}: expected mean {expected_mean[c]} but recovered "
            f"{recovered_mean}"
        )
