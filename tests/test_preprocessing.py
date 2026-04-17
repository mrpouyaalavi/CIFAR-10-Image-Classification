"""
Tests for the preprocessing / transform pipeline.

These tests make sure ``get_transform()`` produces tensors with the exact
shapes and dynamic ranges that the model architectures require. If the
resize target, normalization statistics, or output dtype ever drifts,
inference will silently degrade — so we pin the observable behaviour.
"""

from __future__ import annotations

import math

import pytest
import torch

from model_utils import (
    CIFAR_MEAN,
    CIFAR_STD,
    IMAGENET_MEAN,
    IMAGENET_STD,
    get_transform,
)


def test_custom_cnn_transform_produces_32x32(dummy_pil_image) -> None:
    """Custom CNN expects 3x32x32 float tensors (CIFAR-10 native size)."""
    transform = get_transform("Custom CNN")
    tensor = transform(dummy_pil_image)
    assert isinstance(tensor, torch.Tensor)
    assert tensor.shape == (3, 32, 32)
    assert tensor.dtype == torch.float32


def test_mobilenet_transform_produces_224x224(dummy_pil_image) -> None:
    """MobileNetV2 expects 3x224x224 float tensors (ImageNet input size)."""
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
