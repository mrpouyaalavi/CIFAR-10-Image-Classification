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

<<<<<<< HEAD
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
=======

# ============================================================================
#  Constants — Dataset Statistics & Class Labels
# ============================================================================
# CIFAR-10 contains 60,000 32x32 colour images across 10 mutually exclusive
# classes. The label order below matches torchvision's indexing (0-9).

CLASS_NAMES = (
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
>>>>>>> eb910c8 (Finalize HF app updates and remove tracked checkpoints)
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

<<<<<<< HEAD
_ALL_MODELS = list(_MODEL_MAP.keys())
=======
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


# ============================================================================
#  Model Architectures
# ============================================================================
# Both architectures mirror the definitions used during training in the
# notebook. Weight files (.pth) store only the state_dict (learned parameters),
# so the architecture must be reconstructed identically before loading.
# ============================================================================

class CustomCNN(nn.Module):
    """
    4-Block Convolutional Neural Network — trained from scratch on CIFAR-10.

    Architecture overview:
        - 4 convolutional blocks with progressive channel expansion
          (3 -> 64 -> 128 -> 256 -> 512) to capture increasingly abstract
          features: edges -> textures -> parts -> objects.
        - Each block applies two 3x3 convolutions (dual-conv design) to
          increase the receptive field before spatial downsampling.
        - BatchNorm after each conv layer stabilises training by reducing
          internal covariate shift — the distribution of each layer's input
          stays consistent as earlier weights update.
        - Kaiming He initialisation (implicit in PyTorch's default for
          Conv2d + ReLU) keeps gradient variance roughly constant across
          layers, which is critical in networks this deep.
        - Global Average Pooling (AdaptiveAvgPool2d) in the final block
          collapses each 512-channel feature map to a single scalar,
          eliminating the need for large fully-connected layers and
          reducing overfitting risk.
        - Dropout (0.25 in conv blocks, 0.5 in classifier) randomly zeroes
          activations during training, forcing the network to learn
          redundant representations rather than relying on a few neurons.

    Parameters:
        num_classes: Number of output categories (default: 10 for CIFAR-10).
    """

    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.features = nn.Sequential(
            # Block 1 — Low-level features (edges, colour gradients)
            nn.Conv2d(3, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2), nn.Dropout2d(0.25),

            # Block 2 — Mid-level features (textures, patterns)
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2), nn.Dropout2d(0.25),

            # Block 3 — High-level features (object parts)
            nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2), nn.Dropout2d(0.25),

            # Block 4 — Semantic features (whole-object representations)
            nn.Conv2d(256, 512, 3, padding=1), nn.BatchNorm2d(512), nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(512, 256), nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)  # Flatten: (B, 512, 1, 1) -> (B, 512)
        return self.classifier(x)


def build_mobilenetv2(num_classes: int = 10) -> nn.Module:
    """
    Build MobileNetV2 with a frozen ImageNet backbone for transfer learning.

    Transfer learning strategy:
        MobileNetV2 was pretrained on ImageNet (1.2M images, 1000 classes).
        Its convolutional backbone already encodes powerful, general-purpose
        visual features — edge detectors, texture recognisers, part detectors —
        that transfer well to new image domains.

        By freezing the backbone (requires_grad=False), we prevent catastrophic
        forgetting and train only a lightweight classifier head (12,810 params)
        to map ImageNet features to CIFAR-10's 10 classes.

    Why MobileNetV2:
        - Depthwise separable convolutions reduce computation by ~8-9x
          compared to standard convolutions, making it practical for
          inference on edge devices and mobile platforms.
        - Inverted residual blocks with linear bottlenecks preserve
          information flow through the network.

    Parameters:
        num_classes: Number of output categories (default: 10 for CIFAR-10).

    Returns:
        A MobileNetV2 model with frozen backbone and trainable classifier head.
    """
    from torchvision.models import mobilenet_v2, MobileNet_V2_Weights

    model = mobilenet_v2(weights=MobileNet_V2_Weights.IMAGENET1K_V1)

    # Freeze all backbone parameters — these already encode rich visual features
    for param in model.parameters():
        param.requires_grad = False

    # Replace the 1000-class ImageNet head with a 10-class CIFAR-10 head
    model.classifier = nn.Sequential(
        nn.Dropout(0.2),
        nn.Linear(model.last_channel, num_classes),
    )
    return model


# ============================================================================
#  Device Selection
# ============================================================================

def select_device():
    """
    Auto-detect the best available compute device.

    Priority order:
        1. CUDA  — NVIDIA GPU (fastest for batch inference)
        2. MPS   — Apple Silicon GPU (Metal Performance Shaders)
        3. CPU   — Fallback (always available)
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# ============================================================================
#  Model Loading
# ============================================================================

def load_model(name: str, device: torch.device) -> nn.Module:
    """
    Load a trained model from disk by searching standard checkpoint locations.

    The function searches multiple directories (checkpoints/, models/, ./) and
    multiple naming conventions to handle checkpoints saved at different stages
    of experimentation. Only the state_dict is loaded (weights_only=True) to
    avoid running arbitrary code from untrusted .pth files.

    Parameters:
        name:   Model identifier — "custom_cnn" or "mobilenet".
        device: Target device for tensor placement.

    Returns:
        The loaded model in inference mode, ready for prediction.

    Raises:
        FileNotFoundError: If no checkpoint file is found for the given model.
    """
    search_dirs = ["checkpoints", "models", "."]

    if name == "custom_cnn":
        model = CustomCNN(num_classes=10)
        candidates = [
            "custom_cnn_best.pth", "custom_cnn_final.pth",
            "custom_cnn_model.pth", "custom_cnn.pth",
            "custom_cnn_best_cpufast.pth",
        ]
    elif name == "mobilenet":
        model = build_mobilenetv2(num_classes=10)
        candidates = [
            "mobilenetv2_best.pth", "mobilenetv2_final.pth",
            "mobilenet_model.pth", "mobilenet.pth",
        ]
    else:
        raise ValueError(f"Unknown model: {name}")

    for d in search_dirs:
        for c in candidates:
            p = os.path.join(d, c)
            if os.path.isfile(p):
                state = torch.load(p, map_location=device, weights_only=True)
                model.load_state_dict(state)
                model.to(device).eval()
                print(f"✓ Loaded {name} from {p}")
                return model

    raise FileNotFoundError(f"No checkpoint found for {name}")


# ============================================================================
#  Image Preprocessing Transforms
# ============================================================================

def get_transforms(model_name: str):
    """
    Return the appropriate preprocessing pipeline for each model.

    Image preprocessing is critical because neural networks are sensitive to
    the statistical distribution of their inputs. Each model expects inputs
    normalised to the same distribution it was trained on.

    Custom CNN pipeline:
        1. Resize to 32x32 — the native CIFAR-10 resolution this model was
           designed for. No upscaling overhead.
        2. Convert to tensor — pixel values go from [0, 255] uint8 to
           [0.0, 1.0] float32, which is required for gradient computation.
        3. Normalise with CIFAR-10 statistics — centres each channel to
           zero mean and unit variance using dataset-specific statistics.

    MobileNetV2 pipeline:
        1. Resize to 224x224 — MobileNetV2 was trained on ImageNet at this
           resolution. Its convolutional filters, stride patterns, and
           receptive fields are all calibrated for 224x224 inputs. Feeding
           32x32 images directly would cause feature maps to shrink to
           near-zero spatial dimensions too early in the network.
        2. Convert to tensor — same float32 conversion.
        3. Normalise with ImageNet statistics — must match the distribution
           the backbone was pretrained on to avoid covariate shift.

    Parameters:
        model_name: "custom_cnn" or "mobilenet".

    Returns:
        A torchvision.transforms.Compose pipeline.
    """
    if model_name == "custom_cnn":
        return transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
        ])
    else:
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])


# ============================================================================
#  Single-Image Prediction
# ============================================================================

def predict_single(
    model: nn.Module,
    image: Image.Image,
    transform,
    device: torch.device,
    top_k: int = 3,
):
    """
    Predict class probabilities for a single PIL image.

    Inference pipeline:
        1. Apply the model-specific preprocessing transform.
        2. Add a batch dimension (unsqueeze) — PyTorch expects (B, C, H, W).
        3. Forward pass through the network to produce raw logits.
        4. Apply softmax to convert logits into a probability distribution
           (all values in [0, 1], summing to 1.0).
        5. Return the top-k predictions sorted by confidence.

    The torch.no_grad() context manager disables gradient tracking during
    inference, reducing memory usage and speeding up computation since
    backpropagation is not needed.

    Parameters:
        model:     Trained model in inference mode.
        image:     Input image as a PIL Image (any size — resizing is handled
                   by the transform).
        transform: Preprocessing pipeline matching the model's expected input.
        device:    Compute device (cuda / mps / cpu).
        top_k:     Number of top predictions to return (default: 3).

    Returns:
        A list of (class_name, confidence_percentage) tuples, sorted by
        descending confidence.
    """
    model.eval()
    tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(tensor)
        probs = F.softmax(logits, dim=1)[0]

    top_probs, top_indices = probs.topk(top_k)
    results = []
    for prob, idx in zip(top_probs, top_indices):
        results.append((CLASS_NAMES[idx.item()], prob.item() * 100))
    return results
>>>>>>> eb910c8 (Finalize HF app updates and remove tracked checkpoints)


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

<<<<<<< HEAD
    # ── Misc ──────────────────────────────────────────────────────
    parser.add_argument("--top-k", type=int, default=5,
                        help="Number of top predictions to show (default: 5)")
    parser.add_argument("--save", type=str, default=None,
                        help="Path to save the prediction visualisation figure")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for --test-samples mode (default: 42)")

    args = parser.parse_args()
=======
>>>>>>> eb910c8 (Finalize HF app updates and remove tracked checkpoints)
    device = select_device()
    print(f"Device: {device}\n")

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
<<<<<<< HEAD
    for cli_name, model in loaded_models.items():
        registry_name = _MODEL_MAP[cli_name]
        preds_list = [
            predict(model, img, registry_name, device, top_k=args.top_k)
            for img in images
        ]
        all_results[pretty_model_name(registry_name)] = preds_list
=======
    for model_name, model in models.items():
        transform = get_transforms(model_name)
        preds_list = []
        for img in images:
            preds = predict_single(model, img, transform, device, top_k=args.top_k)
            preds_list.append(preds)
        all_results[model_name.replace("_", " ").title()] = preds_list
>>>>>>> eb910c8 (Finalize HF app updates and remove tracked checkpoints)

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
