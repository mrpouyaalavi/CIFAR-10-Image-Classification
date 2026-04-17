"""
Inference helper tests — ``predict()`` end-to-end contract.

These tests build a freshly-initialised model (i.e. random weights) and
run ``predict()`` through the transform + forward pass. We don't assert
anything about which class comes out on top — the weights are random —
only that the public contract of the function holds:

* returns a list of ``(str, float)`` tuples
* length equals ``top_k``
* confidences are percentages in [0, 100]
* confidences sum to <= 100 (soft cap because top_k may be < num_classes)
* every class name belongs to ``CLASS_NAMES``
* top_k is monotonically non-increasing in confidence
"""

from __future__ import annotations

import pytest
import torch

from model_utils import (
    CLASS_NAMES,
    CustomCNN,
    build_mobilenetv2,
    build_resnet18,
    predict,
)


@pytest.fixture()
def custom_cnn_model(device: torch.device):
    """Randomly-initialised Custom CNN in inference mode."""
    model = CustomCNN(num_classes=10).to(device)
    model.train(False)
    return model


@pytest.fixture()
def mobilenet_model(device: torch.device):
    """Real MobileNetV2 backbone with a randomly-initialised head."""
    model = build_mobilenetv2(num_classes=10).to(device)
    model.train(False)
    return model


@pytest.fixture()
def resnet18_model(device: torch.device):
    """Real ResNet-18 backbone with a randomly-initialised FC head."""
    model = build_resnet18(num_classes=10).to(device)
    model.train(False)
    return model


def test_predict_custom_cnn_contract(
    custom_cnn_model, dummy_pil_image, device: torch.device
) -> None:
    preds = predict(
        custom_cnn_model, dummy_pil_image, "Custom CNN", device, top_k=5,
    )
    assert isinstance(preds, list)
    assert len(preds) == 5
    for class_name, confidence in preds:
        assert class_name in CLASS_NAMES
        assert 0.0 <= confidence <= 100.0
    # Top-5 softmax probabilities sum <= 100%
    total = sum(c for _, c in preds)
    assert total <= 100.0 + 1e-4


def test_predict_mobilenet_contract(
    mobilenet_model, dummy_pil_image, device: torch.device
) -> None:
    preds = predict(
        mobilenet_model, dummy_pil_image, "MobileNetV2", device, top_k=3,
    )
    assert len(preds) == 3
    # Sanity: confidences must be non-increasing (top-k is sorted desc)
    confidences = [c for _, c in preds]
    assert confidences == sorted(confidences, reverse=True)


def test_predict_top_k_one(
    custom_cnn_model, dummy_pil_image, device: torch.device
) -> None:
    preds = predict(
        custom_cnn_model, dummy_pil_image, "Custom CNN", device, top_k=1,
    )
    assert len(preds) == 1
    cls, conf = preds[0]
    assert cls in CLASS_NAMES
    assert 0.0 <= conf <= 100.0


def test_predict_resnet18_contract(
    resnet18_model, dummy_pil_image, device: torch.device
) -> None:
    """ResNet-18 predict() must satisfy the same public contract as the others.

    We also verify that confidences are non-increasing so we catch any
    regression where the top-k sort order breaks for a 224×224 input path.
    """
    preds = predict(
        resnet18_model, dummy_pil_image, "ResNet-18", device, top_k=5,
    )
    assert isinstance(preds, list)
    assert len(preds) == 5
    for class_name, confidence in preds:
        assert class_name in CLASS_NAMES
        assert 0.0 <= confidence <= 100.0
    confidences = [c for _, c in preds]
    assert confidences == sorted(confidences, reverse=True)


def test_predict_top_k_all(
    custom_cnn_model, dummy_pil_image, device: torch.device
) -> None:
    """Top-10 == all classes; the confidences must sum to ~100%."""
    preds = predict(
        custom_cnn_model, dummy_pil_image, "Custom CNN", device, top_k=10,
    )
    assert len(preds) == 10
    assert {cls for cls, _ in preds} == set(CLASS_NAMES)
    total = sum(c for _, c in preds)
    assert abs(total - 100.0) < 1e-3
