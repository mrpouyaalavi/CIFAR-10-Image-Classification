"""
CIFAR-10 Image Classification — Gradio Demo (Hugging Face Spaces)
==================================================================

A polished, portfolio-ready Gradio app that showcases a comparative
deep-learning study on CIFAR-10: Custom CNN (trained from scratch)
vs MobileNetV2 (transfer learning with frozen ImageNet backbone).

Deployed on Hugging Face Spaces.  The original Streamlit app remains
live at https://cifar10-pouyaalavi.streamlit.app/ as a landing page.

Author : Pouya Alavi  (pouya@pouyaalavi.dev)
License: MIT
"""

from __future__ import annotations

import os
import torch
import gradio as gr
import numpy as np
from PIL import Image

from model_utils import (
    CLASS_NAMES,
    list_available_models,
    load_model_by_name,
    predict,
    select_device,
)
from benchmark_data import (
    BENCHMARK_METRICS,
    CONFUSION_PAIRS,
    CONVERGENCE_HISTORY,
    TRAINING_CONFIG,
)

# ============================================================================
#  Global Setup
# ============================================================================

DEVICE = select_device(verbose=True)

# Cache loaded models so we only pay the cost once per architecture.
_model_cache: dict[str, torch.nn.Module] = {}


def _get_model(name: str) -> torch.nn.Module:
    if name not in _model_cache:
        _model_cache[name] = load_model_by_name(name, DEVICE)
    return _model_cache[name]


# Pre-load available model names at startup.
AVAILABLE_MODELS = list_available_models()


# ============================================================================
#  Example images — pull a few from the CIFAR-10 test set
# ============================================================================

def _prepare_examples(n: int = 6) -> list[str]:
    """Save a handful of CIFAR-10 test images as PNGs and return their paths."""
    examples_dir = os.path.join(os.path.dirname(__file__), "examples")
    # If already prepared, just return existing files.
    if os.path.isdir(examples_dir) and len(os.listdir(examples_dir)) >= n:
        paths = sorted(
            os.path.join(examples_dir, f)
            for f in os.listdir(examples_dir)
            if f.endswith(".png")
        )
        return paths[:n]

    os.makedirs(examples_dir, exist_ok=True)

    try:
        import torchvision
        import torchvision.transforms as transforms

        dataset = torchvision.datasets.CIFAR10(
            root="./data", train=False, download=True,
            transform=transforms.ToTensor(),
        )
        # Pick one image per class for variety.
        seen_classes: set[int] = set()
        saved: list[str] = []
        for img_tensor, label in dataset:
            if label in seen_classes:
                continue
            seen_classes.add(label)
            pil = transforms.ToPILImage()(img_tensor)
            # Upscale from 32x32 so examples look decent in the UI.
            pil = pil.resize((128, 128), Image.NEAREST)
            fname = f"{CLASS_NAMES[label]}.png"
            path = os.path.join(examples_dir, fname)
            pil.save(path)
            saved.append(path)
            if len(saved) >= n:
                break
        return sorted(saved)
    except Exception:
        return []


EXAMPLE_IMAGES = _prepare_examples()


# ============================================================================
#  Prediction Logic
# ============================================================================

def classify(image: Image.Image | None, model_name: str) -> dict[str, float]:
    """Run inference on a single image using the selected model.

    Returns a dict mapping class names to confidence percentages,
    which Gradio's gr.Label component renders as a leaderboard.
    """
    if image is None:
        return {}

    image = image.convert("RGB")
    model = _get_model(model_name)
    results = predict(model, image, model_name, DEVICE, top_k=10)
    # Gradio Label expects {label: confidence_fraction}
    return {cls: round(conf / 100.0, 4) for cls, conf in results}


def compare_models(image: Image.Image | None) -> tuple[dict, dict]:
    """Run inference with all available models for side-by-side comparison."""
    if image is None:
        return {}, {}

    image = image.convert("RGB")
    outputs = []
    for name in AVAILABLE_MODELS[:2]:  # Custom CNN and MobileNetV2
        model = _get_model(name)
        results = predict(model, image, name, DEVICE, top_k=10)
        outputs.append({cls: round(conf / 100.0, 4) for cls, conf in results})

    # Pad if fewer than 2 models available.
    while len(outputs) < 2:
        outputs.append({})

    return outputs[0], outputs[1]


# ============================================================================
#  Model Comparison Table (Markdown)
# ============================================================================

def _build_comparison_md() -> str:
    """Build a Markdown table comparing all models in the benchmark."""
    rows = []
    for key, m in BENCHMARK_METRICS.items():
        badge = "**deployed**" if m["available"] else "study only"
        rows.append(
            f"| {m['display_name']} | {m['test_accuracy']:.2f}% "
            f"| {m['trainable_params']:,} | {m['total_params']:,} "
            f"| {m['size_mb']:.1f} MB | {m['latency_ms']:.1f} ms "
            f"| {m['strategy']} | {badge} |"
        )
    header = (
        "| Model | Accuracy | Trainable Params | Total Params "
        "| Size | Latency (CPU) | Strategy | Status |\n"
        "|-------|----------|-----------------|-------------"
        "|------|---------------|----------|--------|\n"
    )
    return header + "\n".join(rows)


def _build_confusion_pairs_md() -> str:
    """Render the top-5 confusion pair reduction table."""
    header = (
        "| Class Pair | Custom CNN Errors | MobileNetV2 Errors | Reduction |\n"
        "|------------|-------------------|--------------------|-----------|\n"
    )
    rows = []
    for p in CONFUSION_PAIRS:
        rows.append(
            f"| {p['pair']} | {p['before']} | {p['after']} "
            f"| {p['reduction_pct']}% |"
        )
    return header + "\n".join(rows)


# ============================================================================
#  Gradio UI
# ============================================================================

TITLE = "CIFAR-10 Image Classification"
DESCRIPTION = (
    "A comparative deep-learning study: **Custom CNN** (trained from scratch) "
    "vs **MobileNetV2** (transfer learning with frozen ImageNet backbone). "
    "Upload an image or click an example below to classify it into one of 10 categories: "
    "airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck."
)

CSS = """
.main-title { text-align: center; margin-bottom: 0.5em; }
.subtitle { text-align: center; color: #666; font-size: 1.05em; margin-bottom: 1.5em; }
footer { display: none !important; }
"""

with gr.Blocks(
    title=f"{TITLE} — Pouya Alavi",
    css=CSS,
    theme=gr.themes.Soft(
        primary_hue="blue",
        secondary_hue="purple",
    ),
) as demo:

    # ── Header ──────────────────────────────────────────────────────
    gr.Markdown(
        f"<h1 class='main-title'>🧠 {TITLE}</h1>\n"
        f"<p class='subtitle'>{DESCRIPTION}</p>"
    )

    with gr.Tabs():
        # ── Tab 1: Live Demo ────────────────────────────────────────
        with gr.TabItem("🔬 Live Demo"):
            gr.Markdown("### Single-Model Prediction")
            with gr.Row():
                with gr.Column(scale=1):
                    img_input = gr.Image(
                        type="pil",
                        label="Upload an image",
                        height=300,
                    )
                    model_dropdown = gr.Dropdown(
                        choices=AVAILABLE_MODELS,
                        value=AVAILABLE_MODELS[0] if AVAILABLE_MODELS else None,
                        label="Select Model",
                    )
                    predict_btn = gr.Button("Classify", variant="primary")

                with gr.Column(scale=1):
                    label_output = gr.Label(
                        num_top_classes=5,
                        label="Predictions",
                    )

            predict_btn.click(
                fn=classify,
                inputs=[img_input, model_dropdown],
                outputs=label_output,
            )

            # Examples
            if EXAMPLE_IMAGES:
                gr.Markdown("### Example Images")
                gr.Examples(
                    examples=[[p] for p in EXAMPLE_IMAGES],
                    inputs=img_input,
                    label="Click an example to load it",
                )

            # ── Side-by-side comparison ─────────────────────────────
            gr.Markdown("---")
            gr.Markdown(
                "### Head-to-Head Comparison\n"
                "Upload an image above and click **Compare Both Models** to see "
                "how Custom CNN and MobileNetV2 differ on the same input."
            )
            compare_btn = gr.Button(
                "Compare Both Models", variant="secondary",
            )
            with gr.Row():
                label_cnn = gr.Label(
                    num_top_classes=5,
                    label=AVAILABLE_MODELS[0] if len(AVAILABLE_MODELS) > 0 else "Model 1",
                )
                label_mobile = gr.Label(
                    num_top_classes=5,
                    label=AVAILABLE_MODELS[1] if len(AVAILABLE_MODELS) > 1 else "Model 2",
                )

            compare_btn.click(
                fn=compare_models,
                inputs=img_input,
                outputs=[label_cnn, label_mobile],
            )

        # ── Tab 2: Model Comparison ─────────────────────────────────
        with gr.TabItem("📊 Model Comparison"):
            gr.Markdown("### Architecture Benchmark")
            gr.Markdown(
                "All models were trained with identical hyperparameters "
                f"(Adam, lr={TRAINING_CONFIG['learning_rate']}, "
                f"{TRAINING_CONFIG['epochs']} epochs, "
                f"batch size {TRAINING_CONFIG['batch_size']}) "
                "on the CIFAR-10 dataset (50,000 training / 10,000 test images)."
            )
            gr.Markdown(_build_comparison_md())

            gr.Markdown("### Key Findings")
            gr.Markdown(
                "- **MobileNetV2** achieves **86.91%** accuracy with only "
                "**12,810 trainable parameters** — thanks to a frozen ImageNet backbone.\n"
                "- **Custom CNN** reaches **48.40%** despite training **2.46M parameters** "
                "from scratch — demonstrating the efficiency gap.\n"
                "- Transfer learning delivers **+38.5 percentage points** higher accuracy "
                "with **192x fewer** trainable parameters.\n"
                "- MobileNetV2 converges in **~3 epochs** vs 15 for the CNN."
            )

            gr.Markdown("### Top-5 Confusion Pair Reductions")
            gr.Markdown(
                "Misclassification counts between the hardest class pairs, "
                "before (Custom CNN) and after (MobileNetV2):"
            )
            gr.Markdown(_build_confusion_pairs_md())

        # ── Tab 3: About ────────────────────────────────────────────
        with gr.TabItem("ℹ️ About"):
            gr.Markdown(
                "## About This Project\n\n"
                "This is an end-to-end deep learning project that designs, trains, "
                "and evaluates multiple CNN architectures on the CIFAR-10 benchmark.\n\n"
                "### Research Question\n"
                "> *How much does a pretrained backbone actually help compared to "
                "training from scratch — when both models share the same training budget?*\n\n"
                "### Dataset\n"
                "**CIFAR-10** — 60,000 32×32 RGB images across 10 classes: "
                "airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck.\n\n"
                "### Training Pipeline\n"
                "- **Augmentation:** RandomCrop, HorizontalFlip, CutOut, MixUp, CutMix\n"
                "- **Optimizer:** Adam (lr=0.001, weight_decay=0.0001)\n"
                "- **Scheduler:** CosineAnnealingLR\n"
                "- **Loss:** CrossEntropyLoss\n"
                "- **Epochs:** 15 (both models, identical budget)\n\n"
                "### Architectures\n"
                "| Architecture | Approach | Trainable Params |\n"
                "|-------------|----------|------------------|\n"
                "| Custom CNN | 4-block CNN from scratch | 2,462,282 |\n"
                "| MobileNetV2 | Frozen ImageNet backbone + linear head | 12,810 |\n"
                "| ResNet-18 | Frozen ImageNet backbone + FC head | 5,130 |\n"
                "| EfficientNet-B0 | Frozen ImageNet backbone + classifier | 12,810 |\n"
                "| ViT-Small | Vision Transformer from scratch | 4,756,746 |\n\n"
                "### Links\n"
                "- **GitHub:** [mrpouyaalavi/CIFAR-10-Image-Classification]"
                "(https://github.com/mrpouyaalavi/CIFAR-10-Image-Classification)\n"
                "- **Streamlit (legacy):** [cifar10-pouyaalavi.streamlit.app]"
                "(https://cifar10-pouyaalavi.streamlit.app/)\n"
                "- **Portfolio:** [pouyaalavi.dev](https://pouyaalavi.dev)\n\n"
                "### Author\n"
                "**Pouya Alavi** — Bachelor of IT student at Macquarie University, "
                "majoring in AI and Web/App Development.\n\n"
                "*Built with PyTorch, Gradio, and Hugging Face Spaces.*"
            )

# ============================================================================
#  Launch
# ============================================================================
# Hugging Face Spaces will call this file directly. The `share=False`
# default is correct for Spaces (the platform handles the public URL).

if __name__ == "__main__":
    demo.launch()
