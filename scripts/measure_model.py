"""
Measure accuracy, confidence calibration, latency, and confusion pairs
for a trained CIFAR-10 model.

Why this script exists
----------------------
After every retraining run we need to refresh ~6 different numbers across
benchmark_data.py, training_metadata.json and the README. Eyeballing them
from Jupyter is error-prone; this script computes them all in one pass so
the downstream updates are purely mechanical transcription.

What it measures
----------------
1. **Top-1 test accuracy** on the full 10 000-image CIFAR-10 test set.
2. **Confidence calibration**: mean top-1 softmax probability separated into
   the correct/incorrect subsets. A healthy model has conf_when_correct in
   the 90s and conf_when_wrong lower (e.g. 60-70%). Our original Custom
   CNN had 50.2% / 34.9% — smoking-gun signs of under-training.
3. **Single-image CPU latency** (median of `--latency-trials` forward
   passes with a warm-up). Only meaningful when the CPU is otherwise idle.
4. **Confusion pair counts** for the five canonical hardest pairs
   (matching the `CONFUSION_PAIRS` list in benchmark_data.py). These feed
   directly into the "per-class error analysis" table.

Usage
-----
    python scripts/measure_model.py --model custom_cnn
    python scripts/measure_model.py --model mobilenet --no-latency
    python scripts/measure_model.py --model resnet18
    python scripts/measure_model.py --model custom_cnn --device cpu
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
import torchvision
import torchvision.transforms as T
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from model_utils import (  # noqa: E402
    CIFAR_MEAN, CIFAR_STD, IMAGENET_MEAN, IMAGENET_STD,
    CLASS_NAMES, CustomCNN, build_mobilenetv2, build_resnet18,
    pretty_model_name, select_device,
)


# The 5 canonical hardest CIFAR-10 pairs (by off-diagonal confusion mass).
# Each tuple (a, b) is matched bidirectionally (a→b and b→a both count).
HARD_PAIRS: list[tuple[str, str]] = [
    ("automobile", "truck"),
    ("airplane",   "ship"),
    ("cat",        "dog"),
    ("dog",        "horse"),
    ("bird",       "deer"),
]

PAIR_DISPLAY = {
    ("automobile", "truck"): "Truck ↔ Automobile",
    ("airplane",   "ship"):  "Ship ↔ Airplane",
    ("cat",        "dog"):   "Cat ↔ Dog",
    ("dog",        "horse"): "Horse ↔ Dog",
    ("bird",       "deer"):  "Bird ↔ Deer",
}

# Maps CLI key → (builder_fn, checkpoint_filename, input_size)
# input_size is the spatial dimension the model requires (square).
_MODEL_REGISTRY: dict[str, tuple] = {
    "custom_cnn": (
        lambda: CustomCNN(num_classes=10),
        "custom_cnn_best.pth",
        32,
    ),
    "mobilenet": (
        lambda: build_mobilenetv2(num_classes=10),
        "mobilenetv2_best.pth",
        224,
    ),
    "resnet18": (
        lambda: build_resnet18(num_classes=10),
        "resnet-18_best.pth",
        224,
    ),
}


def build_model(model_key: str, device: torch.device) -> torch.nn.Module:
    """Load a trained model from its canonical checkpoint path."""
    if model_key not in _MODEL_REGISTRY:
        raise ValueError(
            f"unknown model key {model_key!r}. Choose from: {list(_MODEL_REGISTRY)}"
        )
    builder, ckpt_name, _ = _MODEL_REGISTRY[model_key]
    model = builder()
    ckpt = ROOT / "checkpoints" / ckpt_name
    state = torch.load(ckpt, map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.to(device).train(False)
    return model


def get_test_loader(model_key: str, batch_size: int) -> DataLoader:
    """Test loader with the model-appropriate preprocessing (no augmentation)."""
    _, _, input_size = _MODEL_REGISTRY[model_key]
    if input_size == 32:
        tfm = T.Compose([T.ToTensor(), T.Normalize(CIFAR_MEAN, CIFAR_STD)])
    else:
        # Transfer models expect 224×224 ImageNet-normalised inputs.
        tfm = T.Compose([
            T.Resize((input_size, input_size)),
            T.ToTensor(),
            T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])
    ds = torchvision.datasets.CIFAR10(
        root=str(ROOT / "data"), train=False, download=True, transform=tfm,
    )
    return DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)


def measure_accuracy_and_calibration(
    model: torch.nn.Module, loader: DataLoader, device: torch.device,
) -> dict:
    """Single pass: accuracy, per-class accuracy, confidence buckets, confusion pairs."""
    total = 0
    correct = 0
    conf_correct_sum = 0.0
    conf_wrong_sum = 0.0
    n_correct = 0
    n_wrong = 0
    per_class_correct = [0] * 10
    per_class_total = [0] * 10
    # 10×10 confusion matrix (rows = true, cols = pred)
    confusion = [[0] * 10 for _ in range(10)]

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            logits = model(x)
            probs = F.softmax(logits, dim=1)
            top_conf, pred = probs.max(dim=1)
            is_correct = (pred == y)

            total += y.numel()
            correct += is_correct.sum().item()

            # Confidence buckets
            conf_correct_sum += top_conf[is_correct].sum().item()
            conf_wrong_sum += top_conf[~is_correct].sum().item()
            n_correct += is_correct.sum().item()
            n_wrong += (~is_correct).sum().item()

            # Per-class accuracy + confusion matrix
            for yi, pi in zip(y.tolist(), pred.tolist()):
                per_class_total[yi] += 1
                if yi == pi:
                    per_class_correct[yi] += 1
                confusion[yi][pi] += 1

    accuracy = 100.0 * correct / total
    per_class_acc = {
        CLASS_NAMES[i]: round(100.0 * per_class_correct[i] / per_class_total[i], 2)
        for i in range(10)
    }

    # Bidirectional confusion pair counts
    name_to_idx = {name: i for i, name in enumerate(CLASS_NAMES)}
    pair_counts: list[dict] = []
    for a, b in HARD_PAIRS:
        ai, bi = name_to_idx[a], name_to_idx[b]
        count = confusion[ai][bi] + confusion[bi][ai]
        pair_counts.append({
            "pair": PAIR_DISPLAY[(a, b)],
            "classes": [a, b],
            "count": count,
        })

    return {
        "accuracy_pct": round(accuracy, 2),
        "correct": correct,
        "total": total,
        "conf_when_correct_pct": round(100.0 * conf_correct_sum / max(n_correct, 1), 2),
        "conf_when_wrong_pct": round(100.0 * conf_wrong_sum / max(n_wrong, 1), 2),
        "per_class_accuracy": per_class_acc,
        "confusion_pairs": pair_counts,
    }


def measure_latency(
    model: torch.nn.Module, model_key: str, device: torch.device, trials: int,
) -> dict:
    """Median single-image CPU latency. Warm-up first to stabilise measurements."""
    _, _, input_size = _MODEL_REGISTRY[model_key]
    x = torch.randn(1, 3, input_size, input_size, device=device)

    # Warm-up: first few forward passes allocate memory and compile kernels.
    with torch.no_grad():
        for _ in range(10):
            model(x)

    samples: list[float] = []
    with torch.no_grad():
        for _ in range(trials):
            t0 = time.perf_counter()
            model(x)
            # On MPS/CUDA we need to sync to get real wall-clock; on CPU it's immediate.
            if device.type == "cuda":
                torch.cuda.synchronize()
            elif device.type == "mps":
                torch.mps.synchronize()
            samples.append((time.perf_counter() - t0) * 1000.0)  # ms

    samples.sort()
    return {
        "trials": trials,
        "input_size": input_size,
        "median_ms": round(statistics.median(samples), 3),
        "mean_ms": round(statistics.mean(samples), 3),
        "p95_ms": round(samples[int(0.95 * len(samples))], 3),
        "min_ms": round(samples[0], 3),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        choices=tuple(_MODEL_REGISTRY.keys()),
        required=True,
        help="Which model checkpoint to measure.",
    )
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"),
                        default="auto")
    parser.add_argument("--no-latency", action="store_true",
                        help="Skip latency measurement (useful if CPU is busy).")
    parser.add_argument("--latency-trials", type=int, default=50)
    parser.add_argument("--json-out", type=Path, default=None,
                        help="Optional path to dump the result JSON.")
    args = parser.parse_args()

    if args.device == "auto":
        device = select_device(verbose=False)
    else:
        device = torch.device(args.device)

    print(f"Model:  {pretty_model_name(args.model)}")
    print(f"Device: {device}")
    print()

    model = build_model(args.model, device)
    loader = get_test_loader(args.model, args.batch_size)

    print("Measuring accuracy + confidence + confusion pairs ...")
    t0 = time.time()
    metrics = measure_accuracy_and_calibration(model, loader, device)
    print(f"  done in {time.time() - t0:.1f}s")
    print()

    print(f"Accuracy:            {metrics['accuracy_pct']:.2f}%  "
          f"({metrics['correct']:,}/{metrics['total']:,})")
    print(f"Conf when correct:   {metrics['conf_when_correct_pct']:.2f}%")
    print(f"Conf when wrong:     {metrics['conf_when_wrong_pct']:.2f}%")
    gap = metrics['conf_when_correct_pct'] - metrics['conf_when_wrong_pct']
    print(f"Calibration gap:     {gap:+.2f} pts  (healthy models: 20-40+)")
    print()

    print("Per-class accuracy:")
    for name, acc in metrics["per_class_accuracy"].items():
        print(f"  {name:12s} {acc:6.2f}%")
    print()

    print("Confusion pairs (bidirectional off-diagonal counts):")
    for p in metrics["confusion_pairs"]:
        print(f"  {p['pair']:<22s} {p['count']:4d}")
    print()

    latency_result = None
    if not args.no_latency:
        print(f"Measuring latency on {device} ({args.latency_trials} trials) ...")
        latency_result = measure_latency(model, args.model, device, args.latency_trials)
        print(f"  median: {latency_result['median_ms']:.3f} ms  "
              f"(mean {latency_result['mean_ms']:.3f}  "
              f"p95 {latency_result['p95_ms']:.3f})")
        fps = 1000.0 / latency_result['median_ms'] if latency_result['median_ms'] else 0
        print(f"  throughput: {fps:.0f} fps")
        print()

    if args.json_out:
        out = {
            "model": pretty_model_name(args.model),
            "device": str(device),
            "metrics": metrics,
            "latency": latency_result,
        }
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(out, indent=2))
        print(f"JSON written: {args.json_out}")


if __name__ == "__main__":
    main()
