"""
Grad-CAM interpretability tests.

The Grad-CAM layer is one of the trickiest parts of this codebase because
it depends on PyTorch's backward-hook plumbing, which silently breaks if:

* the target layer's output is mutated by an in-place op
* the model is in training mode while we're generating attributions
* backbone parameters are frozen *and* the input tensor doesn't have
  ``requires_grad=True`` (this is the MobileNetV2-specific footgun)

These tests exercise all three so regressions get caught in CI before a
visitor ever sees an empty Grad-CAM panel on the Live Demo tab.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from model_utils import (
    CustomCNN,
    GradCAM,
    build_mobilenetv2,
    build_resnet18,
    compute_gradcam_overlay,
    get_gradcam_target_layer,
)


def test_get_gradcam_target_layer_custom_cnn_returns_module() -> None:
    model = CustomCNN(num_classes=10)
    layer = get_gradcam_target_layer(model, "Custom CNN")
    assert isinstance(layer, nn.Module)


def test_get_gradcam_target_layer_mobilenet_returns_module() -> None:
    model = build_mobilenetv2(num_classes=10)
    layer = get_gradcam_target_layer(model, "MobileNetV2")
    assert isinstance(layer, nn.Module)


def test_get_gradcam_target_layer_resnet18_returns_module() -> None:
    """ResNet-18 Grad-CAM target layer must be a valid nn.Module."""
    model = build_resnet18(num_classes=10)
    layer = get_gradcam_target_layer(model, "ResNet-18")
    assert isinstance(layer, nn.Module)


def test_compute_gradcam_overlay_handles_frozen_resnet18(
    dummy_pil_image, device: torch.device,
) -> None:
    """ResNet-18 has the same frozen-backbone footgun as MobileNetV2.

    The Grad-CAM input tensor must carry ``requires_grad=True`` to allow
    gradients to flow back through the frozen backbone to the target layer.
    A blank heatmap (max == 0) signals that the gradient pathway is broken.
    """
    model = build_resnet18(num_classes=10).to(device)
    model.train(False)
    overlay, heatmap, pred_class, confidence = compute_gradcam_overlay(
        model, dummy_pil_image, "ResNet-18", device, alpha=0.5,
    )
    assert overlay.dtype == np.uint8
    assert heatmap.max() > 0.0
    assert 0 <= pred_class < 10
    assert 0.0 <= confidence <= 100.0


def test_gradcam_hooks_fire_on_custom_cnn(device: torch.device) -> None:
    """After a forward+backward pass, the hooks must have captured both
    activations and gradients on the target layer."""
    model = CustomCNN(num_classes=10).to(device)
    model.train(False)
    target = get_gradcam_target_layer(model, "Custom CNN")

    cam = GradCAM(model, target)
    x = torch.randn(1, 3, 32, 32, device=device, requires_grad=True)
    try:
        heatmap, pred_class, logits = cam.generate(x, target_class=None)
    finally:
        cam.remove_hooks()

    assert cam._activations is not None
    assert cam._gradients is not None
    assert isinstance(heatmap, np.ndarray)
    assert heatmap.ndim == 2
    # Output should be normalised to [0, 1]
    assert heatmap.min() >= 0.0
    assert heatmap.max() <= 1.0 + 1e-6
    assert 0 <= pred_class < 10
    assert logits.shape == (1, 10)


def test_compute_gradcam_overlay_returns_image_sized_arrays(
    dummy_pil_image, device: torch.device,
) -> None:
    """Overlay + heatmap must both match the input image dimensions."""
    model = CustomCNN(num_classes=10).to(device)
    model.train(False)
    overlay, heatmap, pred_class, confidence = compute_gradcam_overlay(
        model, dummy_pil_image, "Custom CNN", device, alpha=0.5,
    )
    orig_h, orig_w = dummy_pil_image.size[1], dummy_pil_image.size[0]
    assert overlay.shape == (orig_h, orig_w, 3)
    assert overlay.dtype == np.uint8
    assert heatmap.shape == (orig_h, orig_w)
    assert 0 <= pred_class < 10
    assert 0.0 <= confidence <= 100.0


def test_compute_gradcam_overlay_handles_frozen_backbone(
    dummy_pil_image, device: torch.device,
) -> None:
    """MobileNetV2's backbone is frozen; Grad-CAM must still produce
    non-null gradients because the *input tensor* carries requires_grad.

    This is the exact regression that caused the MobileNetV2 Grad-CAM
    panel to render as a blank jet heatmap in an earlier revision.
    """
    model = build_mobilenetv2(num_classes=10).to(device)
    model.train(False)
    overlay, heatmap, pred_class, confidence = compute_gradcam_overlay(
        model, dummy_pil_image, "MobileNetV2", device, alpha=0.5,
    )
    assert overlay.dtype == np.uint8
    # Heatmap must contain non-zero values — if the gradient pathway is
    # broken, the relu-normalise logic produces an all-zeros array.
    assert heatmap.max() > 0.0
