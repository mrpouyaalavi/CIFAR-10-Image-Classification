"""
Model architecture tests.

These are the "does the model even build and run" smoke tests that every
AI/ML project needs in CI. They catch:

* a broken import chain (e.g. torchvision API drift)
* channel or spatial mismatches between conv blocks
* classifier head wired to the wrong input size
* MobileNetV2's transfer-learning head not actually freezing the backbone

They are intentionally cheap — no dataset download, no real checkpoint
load — so the whole matrix runs in a few seconds on a CPU runner.
"""

from __future__ import annotations

import torch

from model_utils import (
    CLASS_NAMES,
    CustomCNN,
    build_mobilenetv2,
    build_resnet18,
)


# ----- CustomCNN ---------------------------------------------------------


def test_custom_cnn_forward_shape(device: torch.device) -> None:
    """A 32x32 batch should yield (batch, num_classes) logits."""
    model = CustomCNN(num_classes=10).to(device)
    model.train(False)
    x = torch.randn(2, 3, 32, 32, device=device)
    with torch.no_grad():
        logits = model(x)
    assert logits.shape == (2, 10)
    assert torch.isfinite(logits).all()


def test_custom_cnn_respects_num_classes(device: torch.device) -> None:
    """The classifier head width must track the ``num_classes`` kwarg.

    Regression guard: an earlier refactor hard-coded ``nn.Linear(256, 10)``
    in the classifier, which silently ignored a non-default ``num_classes``.
    """
    model = CustomCNN(num_classes=7).to(device)
    model.train(False)
    x = torch.randn(1, 3, 32, 32, device=device)
    with torch.no_grad():
        logits = model(x)
    assert logits.shape == (1, 7)


def test_custom_cnn_has_features_and_classifier(device: torch.device) -> None:
    """The ``_remap_legacy_keys()`` helper assumes both submodules exist."""
    model = CustomCNN(num_classes=10).to(device)
    assert hasattr(model, "features")
    assert hasattr(model, "classifier")


# ----- MobileNetV2 -------------------------------------------------------


def test_mobilenet_forward_shape(device: torch.device) -> None:
    model = build_mobilenetv2(num_classes=10).to(device)
    model.train(False)
    x = torch.randn(1, 3, 224, 224, device=device)
    with torch.no_grad():
        logits = model(x)
    assert logits.shape == (1, 10)
    assert torch.isfinite(logits).all()


def test_mobilenet_backbone_is_frozen() -> None:
    """Only the classifier head should have ``requires_grad=True``.

    The whole transfer-learning story in this project hinges on freezing
    the ImageNet backbone. If someone ever flips that off this test
    fails loudly — which is exactly what we want because the published
    86.91% accuracy figure depends on it.
    """
    model = build_mobilenetv2(num_classes=10)
    trainable = [p for p in model.parameters() if p.requires_grad]
    # The head is ``nn.Sequential(Dropout, Linear)`` — Linear has 2 params
    # (weight + bias), Dropout has none. So exactly 2 trainable tensors.
    assert len(trainable) == 2
    # And no parameter outside model.classifier should be trainable.
    for name, param in model.named_parameters():
        if param.requires_grad:
            assert name.startswith("classifier."), (
                f"Backbone parameter {name!r} is trainable - the freeze broke."
            )


def test_mobilenet_trainable_param_count_matches_benchmark() -> None:
    """The head should contain exactly 12 810 trainable parameters.

    This matches ``BENCHMARK_METRICS['MobileNetV2']['trainable_params']``.
    If the number ever changes, the benchmark table is stale (or the
    torchvision MobileNetV2 spec has shifted under us).
    """
    from benchmark_data import BENCHMARK_METRICS

    model = build_mobilenetv2(num_classes=10)
    actual = sum(p.numel() for p in model.parameters() if p.requires_grad)
    expected = BENCHMARK_METRICS["MobileNetV2"]["trainable_params"]
    assert actual == expected


# ----- ResNet-18 ---------------------------------------------------------


def test_resnet18_forward_shape(device: torch.device) -> None:
    """ResNet-18 expects 224×224 input and outputs (batch, 10) logits."""
    model = build_resnet18(num_classes=10).to(device)
    model.train(False)
    x = torch.randn(1, 3, 224, 224, device=device)
    with torch.no_grad():
        logits = model(x)
    assert logits.shape == (1, 10)
    assert torch.isfinite(logits).all()


def test_resnet18_backbone_is_frozen() -> None:
    """Only the final FC layer should require gradients.

    ResNet-18's transfer-learning story mirrors MobileNetV2: the entire
    pretrained backbone is frozen and only the replacement FC head is
    trained. The 87.48% published accuracy depends on this freeze.
    """
    model = build_resnet18(num_classes=10)
    for name, param in model.named_parameters():
        if param.requires_grad:
            assert name.startswith("fc."), (
                f"Backbone parameter {name!r} is trainable — the freeze broke."
            )


def test_resnet18_trainable_param_count_matches_benchmark() -> None:
    """The FC head must contain exactly 5 130 trainable parameters.

    ResNet-18's penultimate layer is 512-wide; the head is ``Linear(512, 10)``:
        weight: 512 × 10 = 5 120
        bias:            10
        total:        5 130

    This matches ``BENCHMARK_METRICS['ResNet-18']['trainable_params']``.
    """
    from benchmark_data import BENCHMARK_METRICS

    model = build_resnet18(num_classes=10)
    actual = sum(p.numel() for p in model.parameters() if p.requires_grad)
    expected = BENCHMARK_METRICS["ResNet-18"]["trainable_params"]
    assert actual == expected


# ----- Class names -------------------------------------------------------


def test_class_names_are_canonical_cifar10() -> None:
    """CIFAR-10 has exactly these ten classes in exactly this order.

    Any drift here breaks the mapping between model logits and the human
    labels surfaced in the UI, CLI, and Grad-CAM overlays.
    """
    assert len(CLASS_NAMES) == 10
    assert CLASS_NAMES == (
        "airplane", "automobile", "bird", "cat", "deer",
        "dog", "frog", "horse", "ship", "truck",
    )
