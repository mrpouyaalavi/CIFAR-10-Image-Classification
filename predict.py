"""
CIFAR-10 Inference Script
=========================

A command-line tool for classifying images using the trained CIFAR-10 models:
  - Custom CNN   — trained from scratch
  - MobileNetV2  — transfer learning (frozen ImageNet backbone)
  - ResNet-18    — transfer learning (frozen ImageNet backbone)

Three input modes:
  1. Single image    — classify one image from disk
  2. Directory       — batch-classify all images in a folder
  3. Test samples    — randomly sample from the CIFAR-10 test split

Each mode prints a top-k confidence ranking per model and optionally saves a
matplotlib visualisation with confidence bar charts.

Usage Examples:
    python predict.py --image photo.png --model mobilenet
    python predict.py --image-dir ./photos/ --model all --save results/out.png
    python predict.py --test-samples 10 --model all

Author:  Pouya Alavi Naeini
License: MIT
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt

# All model architectures, loading, preprocessing, and inference live in
# model_utils — the single source of truth for this repo. predict.py is
# exclusively responsible for CLI argument handling and visualisation.
from model_utils import (
    CLASS_NAMES,
    get_transform,
    load_model_by_name,
    predict,
    pretty_model_name,
    select_device,
)


# ── CLI name → MODEL_REGISTRY key mapping ────────────────────────────────────
#
# CLI names are lowercase/snake_case because that is the Unix convention for
# flags.  MODEL_REGISTRY keys use the display-name style ("Custom CNN") that
# appears in the Gradio demo and benchmark tables.  The map is the only place
# this translation lives — never duplicated elsewhere in this file.

_MODEL_MAP: dict[str, str] = {
    "custom_cnn": "Custom CNN",
    "mobilenet":  "MobileNetV2",
    "resnet18":   "ResNet-18",
}

_ALL_MODELS = list(_MODEL_MAP.keys())


# ============================================================================
#  Prediction Visualisation
# ============================================================================

def visualize_predictions(
    images: list[Image.Image],
    all_results: dict[str, list],
    labels: list[str] | None = None,
    save_path: str | None = None,
) -> None:
    """
    Render a grid of images alongside horizontal confidence bar charts.

    Layout: each row contains the input image followed by one bar chart per
    model.  Bar colours indicate correctness when ground-truth labels are
    available (green = correct, red = incorrect, grey = lower-ranked).

    Parameters
    ----------
    images:      List of PIL images to display.
    all_results: Dict mapping model display names to their prediction lists.
                 Each list entry is a list of (class_name, confidence_pct) tuples.
    labels:      Optional ground-truth class names for colour-coded accuracy.
    save_path:   Optional file path to save the figure (PNG, 150 DPI).
    """
    n = len(images)
    if n == 0:
        return

    n_models = len(all_results)
    fig, axes = plt.subplots(n, 1 + n_models, figsize=(4 * (1 + n_models), 3 * n))
    if n == 1:
        axes = axes[np.newaxis, :]

    for row in range(n):
        # ── Column 0: original image ──────────────────────────────
        axes[row, 0].imshow(images[row])
        title = f"Image {row}"
        if labels:
            title += f"\n(True: {labels[row]})"
        axes[row, 0].set_title(title, fontsize=10)
        axes[row, 0].axis("off")

        # ── Columns 1..N: confidence bars, one per model ──────────
        for col, (model_name, preds_list) in enumerate(all_results.items(), start=1):
            preds = preds_list[row]
            names = [p[0] for p in preds]
            probs = [p[1] for p in preds]

            # Green top-1 bar when prediction is correct; red otherwise
            correct = labels and names[0] == labels[row]
            bar_colors = ["#2ecc71" if correct else "#e74c3c"] + ["#95a5a6"] * (len(preds) - 1)

            axes[row, col].barh(range(len(preds)), probs, color=bar_colors)
            axes[row, col].set_yticks(range(len(preds)))
            axes[row, col].set_yticklabels(names, fontsize=9)
            axes[row, col].set_xlim(0, 100)
            axes[row, col].set_xlabel("Confidence (%)", fontsize=8)
            axes[row, col].set_title(model_name, fontsize=10, fontweight="bold")
            axes[row, col].invert_yaxis()

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved → {save_path}")
    plt.show()
    plt.close(fig)


# ============================================================================
#  CLI Entry Point
# ============================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="CIFAR-10 prediction script — Custom CNN, MobileNetV2, ResNet-18",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # ── Input source (mutually exclusive) ────────────────────────
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--image",        type=str, help="Path to a single image file")
    group.add_argument("--image-dir",    type=str, help="Directory of image files")
    group.add_argument("--test-samples", type=int, help="Number of random CIFAR-10 test images")

    # ── Model selection ───────────────────────────────────────────
    parser.add_argument(
        "--model",
        choices=_ALL_MODELS + ["all"],
        default="all",
        help="Model(s) to run. 'all' runs every deployed model (default: all)",
    )

    # ── Misc ──────────────────────────────────────────────────────
    parser.add_argument("--top-k", type=int, default=5,
                        help="Number of top predictions to show (default: 5)")
    parser.add_argument("--save", type=str, default=None,
                        help="Path to save the prediction visualisation figure")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for --test-samples mode (default: 42)")

    args = parser.parse_args()
    device = select_device()

    # ── Load requested models ─────────────────────────────────────
    cli_names = _ALL_MODELS if args.model == "all" else [args.model]

    loaded_models: dict[str, torch.nn.Module] = {}
    for cli_name in cli_names:
        registry_name = _MODEL_MAP[cli_name]
        try:
            loaded_models[cli_name] = load_model_by_name(registry_name, device)
            print(f"✓  Loaded {registry_name}")
        except (FileNotFoundError, KeyError) as exc:
            print(f"⚠  Skipping {registry_name}: {exc}", file=sys.stderr)

    if not loaded_models:
        print("No models could be loaded. Exiting.", file=sys.stderr)
        sys.exit(1)

    # ── Prepare input images ──────────────────────────────────────
    images: list[Image.Image] = []
    labels: list[str] | None = None

    if args.image:
        images.append(Image.open(args.image).convert("RGB"))

    elif args.image_dir:
        exts = {".png", ".jpg", ".jpeg", ".bmp", ".tiff"}
        for fname in sorted(os.listdir(args.image_dir)):
            if os.path.splitext(fname)[1].lower() in exts:
                images.append(
                    Image.open(os.path.join(args.image_dir, fname)).convert("RGB")
                )
        if not images:
            print(f"No images found in {args.image_dir}", file=sys.stderr)
            sys.exit(1)

    elif args.test_samples:
        np.random.seed(args.seed)
        raw_dataset = torchvision.datasets.CIFAR10(
            root="./data", train=False, download=True,
            transform=transforms.ToTensor(),
        )
        indices = np.random.choice(len(raw_dataset), size=args.test_samples, replace=False)
        labels = []
        for idx in indices:
            tensor, label = raw_dataset[idx]
            images.append(transforms.ToPILImage()(tensor))
            labels.append(CLASS_NAMES[label])

    # ── Run inference ─────────────────────────────────────────────
    # all_results maps display name → list-of-predictions (one per image)
    all_results: dict[str, list] = {}
    for cli_name, model in loaded_models.items():
        registry_name = _MODEL_MAP[cli_name]
        preds_list = [
            predict(model, img, registry_name, device, top_k=args.top_k)
            for img in images
        ]
        all_results[pretty_model_name(registry_name)] = preds_list

    # ── Print results ─────────────────────────────────────────────
    for i in range(len(images)):
        print(f"\n{'─' * 44}")
        header = f"  Image {i}"
        if labels:
            header += f"  (True: {labels[i]})"
        print(header)
        for model_display, preds_list in all_results.items():
            print(f"  {model_display}:")
            for cls, prob in preds_list[i]:
                bar = "█" * int(prob / 5)   # 1 block per 5 pp
                print(f"    {cls:>12s}: {prob:5.1f}%  {bar}")

    # ── Visualise ─────────────────────────────────────────────────
    if len(images) <= 16:
        visualize_predictions(images, all_results, labels=labels, save_path=args.save)
    else:
        print(
            f"\nℹ  Visualisation skipped ({len(images)} images > 16-image limit).",
            file=sys.stderr,
        )
        if args.save:
            print("   Re-run with ≤ 16 images to save a figure.", file=sys.stderr)

    print("\n✅  Prediction complete!")


if __name__ == "__main__":
    main()
