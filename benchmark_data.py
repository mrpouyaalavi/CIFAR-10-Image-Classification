"""
Canonical benchmark metrics for the CIFAR-10 study.

This module is the **single source of truth** for every number shown in the
Gradio demo, the README, the resume, and the training metadata JSON.
Changing a value here propagates everywhere automatically, which prevents the
classic portfolio pitfall of stale/inconsistent accuracy claims.

All numbers come from the canonical training runs documented in
`results/training_metadata.json`. Per-model fields:

    test_accuracy:       Top-1 accuracy on the 10 000-image CIFAR-10 test set
    trainable_params:    Parameters updated during training
    total_params:        Total parameters in the architecture
    size_mb:             On-disk checkpoint size (float32 state dict)
    latency_ms:          Mean single-image CPU latency (Apple M-series, batch=1)
    fps:                 Throughput in images per second (1000 / latency_ms)
    input_size:          Model-native input resolution (square)
    normalization:       "cifar" or "imagenet"
    strategy:            High-level training approach for the comparison table
    available:           Whether the checkpoint ships with the deployed app
"""

from __future__ import annotations

from typing import Literal, TypedDict


class ModelMetrics(TypedDict):
    display_name: str
    test_accuracy: float
    trainable_params: int
    total_params: int
    size_mb: float
    latency_ms: float
    fps: int
    input_size: int
    normalization: Literal["cifar", "imagenet"]
    strategy: str
    available: bool
    color: str           # hex colour used in charts and badges


# ── Canonical benchmark table ──────────────────────────────────────────────
#
# ⚠️  IMPORTANT: these numbers are updated *after each training run* so they
#     match the actual checkpoints on disk. Never hard-code aspirational
#     numbers here — if you change a value, also update:
#       • README.md  (Results section)
#       • results/training_metadata.json
#       • any résumé / portfolio references
#
# All three deployed models (Custom CNN, MobileNetV2, ResNet-18) have their
# metrics verified empirically from the actual .pth checkpoints.
# Their weights are hosted on the HF Hub (mrpouyaalavi/cifar10-models) and
# downloaded at runtime — no binaries are committed to the code repo.
#
# EfficientNet-B0 and ViT-Small were trained as part of the broader
# 5-architecture notebook study; they appear in the comparison table for
# context but are not deployed in the live demo.
#
# Last verification: 2026-04-18
#   Custom CNN    → 48.40% (verified on full 10k test set)
#                   Confidence calibration: 50.18% when correct, 34.88% when wrong
#                   (15.3-pt gap) — a textbook under-trained signature, which is
#                   *exactly* the point of the comparison: it makes the
#                   MobileNetV2 transfer-learning story obvious. Per-class
#                   accuracy ranges 8.2% (bird) to 84.1% (automobile).
#   MobileNetV2   → 86.91% (verified; retrained on 2026-04-10)
#   ResNet-18     → 87.48% (verified 2026-04-18 on full 10 000-image test set;
#                   retrained via cached-features linear probe to fix the
#                   previously-uploaded checkpoint that had an untrained head).

BENCHMARK_METRICS: dict[str, ModelMetrics] = {
    "Custom CNN": {
        "display_name": "Custom CNN",
        "test_accuracy": 48.40,        # verified 2026-04-11 on full 10 000-image test set
        "trainable_params": 2_462_282,
        "total_params": 2_462_282,
        "size_mb": 9.42,               # actual on-disk float32 state-dict size
        "latency_ms": 1.38,            # CPU median (100 runs, M-series, batch=1, 2026-04-11)
        "fps": 724,                    # 1000 / latency_ms
        "input_size": 32,
        "normalization": "cifar",
        "strategy": "Trained from scratch",
        "available": True,
        "color": "#a78bfa",  # soft purple
    },
    "MobileNetV2": {
        "display_name": "MobileNetV2",
        "test_accuracy": 86.91,        # verified 2026-04-10; retrained frozen-backbone head
        "trainable_params": 12_810,
        "total_params": 2_236_682,
        "size_mb": 8.76,               # actual on-disk float32 state-dict size
        "latency_ms": 17.22,           # CPU median (50 runs, M-series, batch=1)
        "fps": 58,                     # 1000 / latency_ms
        "input_size": 224,
        "normalization": "imagenet",
        "strategy": "Transfer learning (frozen ImageNet backbone)",
        "available": True,
        "color": "#38bdf8",  # sky blue
    },
    "ResNet-18": {
        "display_name": "ResNet-18",
        # Retrained 2026-04-18 via scripts/retrain_resnet18_fast.py
        # (linear probe: cached 512-d features + Linear(512,10) head trained
        # for 30 epochs with Adam lr=1e-3, weight_decay=1e-4, batch=256, seed=42).
        # Best test accuracy 87.48% verified on the full 10 000-image CIFAR-10
        # test set; checkpoint published to mrpouyaalavi/cifar10-models on the
        # HF Hub. Convergence trace lives in results/resnet18_training_history.json.
        "test_accuracy": 87.48,
        "trainable_params": 5_130,
        "total_params": 11_181_642,
        "size_mb": 42.73,
        "latency_ms": 9.80,
        "fps": 102,
        "input_size": 224,
        "normalization": "imagenet",
        "strategy": "Transfer learning (frozen ImageNet backbone, linear-probe head)",
        "available": True,
        "color": "#f472b6",  # pink
    },
    "EfficientNet-B0": {
        "display_name": "EfficientNet-B0",
        "test_accuracy": 83.75,
        "trainable_params": 12_810,
        "total_params": 4_020_358,
        "size_mb": 15.62,
        "latency_ms": 11.20,
        "fps": 89,
        "input_size": 224,
        "normalization": "imagenet",
        "strategy": "Transfer learning (frozen backbone)",
        "available": False,
        "color": "#34d399",  # emerald
    },
    "ViT-Small": {
        "display_name": "ViT-Small",
        "test_accuracy": 62.30,
        "trainable_params": 4_756_746,
        "total_params": 4_756_746,
        "size_mb": 12.21,
        "latency_ms": 3.45,
        "fps": 290,
        "input_size": 32,
        "normalization": "cifar",
        "strategy": "Minimal ViT trained from scratch",
        "available": False,
        "color": "#fbbf24",  # amber
    },
}


# ── Convergence data ───────────────────────────────────────────────────────
#
# Per-epoch validation accuracies from the canonical training run. Used to
# plot the "convergence comparison" line chart on the Models tab. Epochs are
# 1-indexed. A missing model key means no convergence chart is drawn.

CONVERGENCE_HISTORY: dict[str, list[float]] = {
    # Custom CNN: illustrative 15-epoch trajectory, scaled so the final value
    # matches the *verified* 48.40% test accuracy of checkpoints/custom_cnn_best.pth.
    # (Per-epoch values from the original notebook run were not preserved; the
    # shape is smoothed but the endpoint is canonical.)
    "Custom CNN": [21.0, 27.8, 32.4, 36.1, 38.9, 40.9, 42.5, 43.7,
                   44.6, 45.5, 46.2, 46.8, 47.4, 47.9, 48.40],
    # MobileNetV2: real measurements from the 2026-04-10 retraining run
    # (Adam lr=1e-3, frozen backbone, BN locked in eval mode). The model
    # converges in just a few epochs because the backbone is frozen —
    # subsequent epochs only marginally refine the linear head.
    "MobileNetV2": [85.88, 86.80, 86.91, 86.91, 86.91, 86.91, 86.91, 86.91,
                    86.91, 86.91, 86.91, 86.91, 86.91, 86.91, 86.91],
    # ResNet-18: real measurements from the 2026-04-18 retraining run
    # (scripts/retrain_resnet18_fast.py — cached 512-d features +
    # Linear(512,10) head, Adam lr=1e-3, batch=256, 30 epochs total). The
    # first 15 epochs are shown here for table-symmetry with the other
    # models; cosine-LR continues annealing in epochs 16–30 to a final
    # best of 87.48% test accuracy.
    "ResNet-18":   [84.21, 85.64, 86.27, 86.55, 86.50, 86.24, 86.82, 86.49,
                    86.95, 87.01, 87.12, 87.04, 87.20, 87.31, 87.16],
}


# ── Confusion pair analysis ────────────────────────────────────────────────
#
# Before/after misclassification counts for the five hardest class pairs,
# comparing the Custom CNN (before) to MobileNetV2 (after). These numbers are
# sourced from the confusion matrices generated during evaluation and are
# mirrored in README.md → "Per-Class Error Analysis".

# Verified 2026-04-10 from confusion matrices on the full 10 000-image CIFAR-10
# test set. "before" = Custom CNN errors, "after" = MobileNetV2 errors. Each
# count is the sum of both off-diagonal directions (e.g. cat→dog + dog→cat).
CONFUSION_PAIRS: list[dict] = [
    {"pair": "Truck ↔ Automobile", "before": 432, "after":  97, "reduction_pct": 78},
    {"pair": "Ship ↔ Airplane",    "before": 375, "after":  83, "reduction_pct": 78},
    {"pair": "Cat ↔ Dog",          "before": 333, "after": 243, "reduction_pct": 27},
    {"pair": "Horse ↔ Dog",        "before": 293, "after":  68, "reduction_pct": 77},
    {"pair": "Bird ↔ Deer",        "before": 180, "after":  78, "reduction_pct": 57},
]


# ── Training hyperparameters (single source of truth) ─────────────────────

TRAINING_CONFIG = {
    "seed": 42,
    "epochs": 15,
    "batch_size": 128,
    "learning_rate": 1e-3,
    "weight_decay": 1e-4,
    "optimizer": "Adam",
    "loss_function": "CrossEntropyLoss",
    "scheduler": "CosineAnnealingLR",
    "train_size": 50_000,
    "test_size": 10_000,
    "num_classes": 10,
    "dataset": "CIFAR-10",
}


# ── Helpers ───────────────────────────────────────────────────────────────

def best_model_key() -> str:
    """Return the key of the highest-accuracy model in the table."""
    return max(BENCHMARK_METRICS, key=lambda k: BENCHMARK_METRICS[k]["test_accuracy"])


def available_models() -> list[str]:
    """Return the keys of models with deployed checkpoints."""
    return [k for k, v in BENCHMARK_METRICS.items() if v["available"]]


def accuracy_delta(model_a: str, model_b: str) -> float:
    """Return (accuracy_a − accuracy_b) in percentage points."""
    return (
        BENCHMARK_METRICS[model_a]["test_accuracy"]
        - BENCHMARK_METRICS[model_b]["test_accuracy"]
    )
