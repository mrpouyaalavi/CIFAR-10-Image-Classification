"""
CIFAR-10 Image Classification — Gradio Demo (Hugging Face Spaces)
==================================================================

Three architectures compared on CIFAR-10:
  · Custom CNN       — trained from scratch (48.40 %)
  · MobileNetV2      — frozen ImageNet backbone (86.91 %)
  · ResNet-18        — frozen ImageNet backbone (82.10 %)

Model weights are downloaded from the HF Hub on first use and cached
for the lifetime of the Space container.

Author : Pouya Alavi  (pouya@pouyaalavi.dev)
License: MIT
"""

from __future__ import annotations

import logging
import os
import traceback

import gradio as gr
import torch
from PIL import Image

from benchmark_data import (
    BENCHMARK_METRICS,
    CONFUSION_PAIRS,
    TRAINING_CONFIG,
)
from model_utils import (
    CLASS_NAMES,
    load_model_by_name,
    predict,
    select_device,
)

# ── Logging ──────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="[cifar10] %(levelname)s: %(message)s",
)
log = logging.getLogger(__name__)

# ── Device ───────────────────────────────────────────────────────────────────

try:
    DEVICE = select_device(verbose=True)
    log.info("Device selected: %s", DEVICE)
except Exception:
    DEVICE = torch.device("cpu")
    log.warning("Device selection failed, falling back to CPU:\n%s", traceback.format_exc())

# ── The three models required by the project spec ────────────────────────────
#
# We hard-code this list instead of calling list_available_models() at import
# time.  list_available_models() triggers hf_hub_download() for every model,
# which can fail during the HF Spaces build phase (no network) and cause the
# entire app module to crash before Gradio can register its API — manifesting
# as "No API found".  Lazy loading on first request avoids this entirely.

DEPLOYED_MODELS: list[str] = ["Custom CNN", "MobileNetV2", "ResNet-18"]

# ── Lazy model cache ─────────────────────────────────────────────────────────

_model_cache: dict[str, torch.nn.Module] = {}
_model_errors: dict[str, str] = {}


def _get_model(name: str) -> torch.nn.Module:
    """Load model on first call; return cached instance thereafter."""
    if name in _model_cache:
        return _model_cache[name]

    if name in _model_errors:
        raise RuntimeError(_model_errors[name])

    log.info("Loading model: %s", name)
    try:
        model = load_model_by_name(name, DEVICE)
        _model_cache[name] = model
        log.info("Model loaded OK: %s", name)
        return model
    except Exception as exc:
        msg = f"Failed to load {name}: {exc}"
        _model_errors[name] = msg
        log.error(msg)
        log.debug(traceback.format_exc())
        raise RuntimeError(msg) from exc


# ── Example images ───────────────────────────────────────────────────────────

def _prepare_examples(n: int = 6) -> list[str]:
    """Save CIFAR-10 test images as PNGs and return their paths."""
    examples_dir = os.path.join(os.path.dirname(__file__), "examples")

    if os.path.isdir(examples_dir):
        pngs = sorted(
            os.path.join(examples_dir, f)
            for f in os.listdir(examples_dir)
            if f.endswith(".png")
        )
        if len(pngs) >= n:
            return pngs[:n]

    os.makedirs(examples_dir, exist_ok=True)
    try:
        import torchvision
        import torchvision.transforms as transforms

        dataset = torchvision.datasets.CIFAR10(
            root="./data", train=False, download=True,
            transform=transforms.ToTensor(),
        )
        seen: set[int] = set()
        saved: list[str] = []
        for img_tensor, label in dataset:
            if label in seen:
                continue
            seen.add(label)
            pil = transforms.ToPILImage()(img_tensor).resize((128, 128), Image.NEAREST)
            path = os.path.join(examples_dir, f"{CLASS_NAMES[label]}.png")
            pil.save(path)
            saved.append(path)
            if len(saved) >= n:
                break
        return sorted(saved)
    except Exception:
        log.warning("Could not prepare example images:\n%s", traceback.format_exc())
        return []


EXAMPLE_IMAGES: list[str] = _prepare_examples()


# ── Prediction functions ──────────────────────────────────────────────────────

def classify(image: Image.Image | None, model_name: str) -> dict[str, float]:
    """Single-model inference.  Returns {class: probability} for gr.Label."""
    if image is None:
        return {}
    try:
        model = _get_model(model_name)
        results = predict(model, image.convert("RGB"), model_name, DEVICE, top_k=10)
        return {cls: round(conf / 100.0, 4) for cls, conf in results}
    except Exception as exc:
        log.error("classify() failed for %s: %s", model_name, exc)
        return {"error — check logs": 1.0}


def compare_all_models(
    image: Image.Image | None,
) -> tuple[dict, dict, dict]:
    """Run inference with all three deployed models.

    Always returns exactly three dicts so the three gr.Label outputs are
    always satisfied, regardless of which models loaded successfully.
    """
    empty: dict[str, float] = {}
    if image is None:
        return empty, empty, empty

    rgb = image.convert("RGB")
    outputs: list[dict] = []
    for name in DEPLOYED_MODELS:
        try:
            model = _get_model(name)
            results = predict(model, rgb, name, DEVICE, top_k=10)
            outputs.append({cls: round(conf / 100.0, 4) for cls, conf in results})
        except Exception as exc:
            log.error("compare_all_models() failed for %s: %s", name, exc)
            outputs.append({"error loading model": 1.0})

    # Guarantee exactly 3 outputs (pad if needed — should never happen)
    while len(outputs) < 3:
        outputs.append(empty)

    return outputs[0], outputs[1], outputs[2]


# ── Markdown helpers ──────────────────────────────────────────────────────────

def _comparison_table_md() -> str:
    rows = []
    for _k, m in BENCHMARK_METRICS.items():
        badge = "**live**" if m["available"] else "study only"
        rows.append(
            f"| {m['display_name']} | {m['test_accuracy']:.2f}% "
            f"| {m['trainable_params']:,} | {m['total_params']:,} "
            f"| {m['size_mb']:.1f} MB | {m['latency_ms']:.1f} ms "
            f"| {m['strategy']} | {badge} |"
        )
    header = (
        "| Model | Accuracy | Trainable Params | Total Params "
        "| Size | CPU Latency | Strategy | Status |\n"
        "|-------|----------|-----------------|-------------|"
        "------|------------|----------|--------|\n"
    )
    return header + "\n".join(rows)


def _confusion_pairs_md() -> str:
    header = (
        "| Pair | Custom CNN | MobileNetV2 | Reduction |\n"
        "|------|-----------|-------------|----------|\n"
    )
    rows = [
        f"| {p['pair']} | {p['before']} | {p['after']} | {p['reduction_pct']}% |"
        for p in CONFUSION_PAIRS
    ]
    return header + "\n".join(rows)


# ── Gradio UI ─────────────────────────────────────────────────────────────────

_CSS = """
.title  { text-align: center; }
.sub    { text-align: center; color: #666; font-size: 1rem; margin-bottom: 1rem; }
footer  { display: none !important; }
"""

with gr.Blocks(title="CIFAR-10 — Pouya Alavi", css=_CSS) as demo:

    gr.Markdown(
        "<h1 class='title'>🧠 CIFAR-10 Image Classification</h1>"
        "<p class='sub'>Custom CNN · MobileNetV2 · ResNet-18 — "
        "a controlled comparison of from-scratch vs transfer learning</p>"
    )

    with gr.Tabs():

        # ── Tab 1: Live Demo ──────────────────────────────────────────────
        with gr.TabItem("🔬 Live Demo"):

            with gr.Row():
                with gr.Column(scale=1):
                    img_in = gr.Image(type="pil", label="Upload an image", height=280)
                    model_dd = gr.Dropdown(
                        choices=DEPLOYED_MODELS,
                        value=DEPLOYED_MODELS[0],
                        label="Model",
                    )
                    predict_btn = gr.Button("Classify ▶", variant="primary")

                with gr.Column(scale=1):
                    single_out = gr.Label(num_top_classes=5, label="Predictions")

            predict_btn.click(
                fn=classify,
                inputs=[img_in, model_dd],
                outputs=single_out,
            )

            if EXAMPLE_IMAGES:
                gr.Markdown("#### Example images — click to load")
                gr.Examples(
                    examples=[[p] for p in EXAMPLE_IMAGES],
                    inputs=img_in,
                )

            gr.Markdown("---")
            gr.Markdown(
                "#### Compare All Three Models\n"
                "Load an image above, then click the button to classify it "
                "with Custom CNN, MobileNetV2, and ResNet-18 simultaneously."
            )
            compare_btn = gr.Button("Compare All Models ▶", variant="secondary")

            with gr.Row():
                out_cnn    = gr.Label(num_top_classes=5, label="Custom CNN")
                out_mobile = gr.Label(num_top_classes=5, label="MobileNetV2")
                out_resnet = gr.Label(num_top_classes=5, label="ResNet-18")

            compare_btn.click(
                fn=compare_all_models,
                inputs=img_in,
                outputs=[out_cnn, out_mobile, out_resnet],
            )

        # ── Tab 2: Model Comparison ───────────────────────────────────────
        with gr.TabItem("📊 Model Comparison"):

            gr.Markdown(
                "### Architecture Benchmark\n\n"
                "All models trained with identical settings — "
                f"Adam lr={TRAINING_CONFIG['learning_rate']}, "
                f"{TRAINING_CONFIG['epochs']} epochs, "
                f"batch {TRAINING_CONFIG['batch_size']} — "
                "on the full CIFAR-10 dataset."
            )
            gr.Markdown(_comparison_table_md())

            gr.Markdown(
                "### Key Findings\n"
                "- **MobileNetV2** — 86.91 % accuracy with 12,810 trainable params "
                "(frozen backbone + linear head).\n"
                "- **ResNet-18** — 82.10 % accuracy with only 5,130 trainable params "
                "(fewest of any deployed model).\n"
                "- **Custom CNN** — 48.40 % accuracy with 2.46 M trainable params "
                "(trained from scratch — demonstrates the transfer-learning gap).\n"
                "- MobileNetV2 achieves **+38.5 pp** over the CNN with **192× fewer** "
                "trainable parameters."
            )

            gr.Markdown("### Top-5 Hardest Confusion Pairs")
            gr.Markdown(_confusion_pairs_md())

        # ── Tab 3: About ──────────────────────────────────────────────────
        with gr.TabItem("ℹ️ About"):

            gr.Markdown(
                "## About This Project\n\n"
                "> *How much does a pretrained backbone actually help compared to "
                "training from scratch — when both models share the same training budget?*\n\n"
                "### Dataset\n"
                "CIFAR-10 — 60,000 32×32 RGB images, 10 classes.\n\n"
                "### Training\n"
                "- Augmentation: RandomCrop, HFlip, CutOut, MixUp, CutMix\n"
                "- Optimiser: Adam (lr 0.001, wd 1e-4), CosineAnnealingLR, 15 epochs\n\n"
                "### Deployed Models\n"
                "| Model | Approach | Trainable Params | Accuracy |\n"
                "|-------|----------|-----------------|----------|\n"
                "| Custom CNN | From scratch | 2,462,282 | 48.40 % |\n"
                "| MobileNetV2 | Frozen backbone | 12,810 | 86.91 % |\n"
                "| ResNet-18 | Frozen backbone | 5,130 | 82.10 % |\n\n"
                "### Links\n"
                "- [GitHub](https://github.com/mrpouyaalavi/CIFAR-10-Image-Classification)\n"
                "- [Legacy Streamlit app](https://cifar10-pouyaalavi.streamlit.app/)\n"
                "- [Portfolio](https://pouyaalavi.dev)\n\n"
                "**Pouya Alavi** — BIT student, Macquarie University "
                "(AI & Web/App Development).\n\n"
                "*PyTorch · Gradio · Hugging Face Spaces*"
            )

# ── Entry point ───────────────────────────────────────────────────────────────
# HF Spaces imports this module and looks for the `demo` object.
# Calling demo.launch() here also allows `python app.py` to work locally.

if __name__ == "__main__":
    demo.launch()
