"""
CIFAR-10 Image Classification — Gradio Demo (Hugging Face Spaces)
==================================================================

Three architectures compared on CIFAR-10:
  · Custom CNN   — trained from scratch       (48.40 %)
  · MobileNetV2  — frozen ImageNet backbone   (86.91 %)
  · ResNet-18    — frozen ImageNet backbone   (87.48 %)

Model weights are downloaded from the HF Hub on first use and cached
for the lifetime of the Space container.

Author : Pouya Alavi Naeini  (pouya@pouyaalavi.dev)
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
    available_models,
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

DEPLOYED_MODELS: list[str] = available_models()  # derived from BENCHMARK_METRICS["...available"]

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
            pil = transforms.ToPILImage()(img_tensor).resize((128, 128), Image.BILINEAR)
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
        raise gr.Error(f"Inference failed for {model_name}. Check Space logs for details.") from exc


def compare_all_models(image: Image.Image | None) -> list[dict]:
    """Run inference with every deployed model.

    Returns one dict per entry in ``DEPLOYED_MODELS`` (in order). The Gradio
    Row below is built dynamically with the same length, so adding/removing
    a model in benchmark_data.py automatically reshapes the comparison UI
    — no hardcoded 3-tuple to drift from.
    """
    empty: dict[str, float] = {}
    if image is None:
        return [empty for _ in DEPLOYED_MODELS]

    rgb = image.convert("RGB")
    outputs: list[dict] = []
    for name in DEPLOYED_MODELS:
        try:
            model = _get_model(name)
            results = predict(model, rgb, name, DEVICE, top_k=10)
            outputs.append({cls: round(conf / 100.0, 4) for cls, conf in results})
        except Exception as exc:
            log.error("compare_all_models() failed for %s: %s", name, exc)
            outputs.append({f"⚠ {name} unavailable": 1.0})

    return outputs


# ── Markdown helpers ──────────────────────────────────────────────────────────

def _comparison_table_md() -> str:
    rows = []
    for _k, m in BENCHMARK_METRICS.items():
        badge = "live" if m["available"] else "study only"
        rows.append(
            f"| {m['display_name']} | {m['test_accuracy']:.2f}% "
            f"| {m['trainable_params']:,} | {m['total_params']:,} "
            f"| {m['size_mb']:.2f} MB | {m['latency_ms']:.2f} ms "
            f"| {m['strategy']} | {badge} |"
        )
    header = (
        "| Model | Accuracy | Trainable Params | Total Params "
        "| Size | CPU Latency | Strategy | Status |\n"
        "|-------|----------|-----------------|-------------|"
        "------|------------|----------|--------|\n"
    )
    return header + "\n".join(rows)


def _key_findings_md() -> str:
    """Build the Key Findings bullet list dynamically from BENCHMARK_METRICS."""
    bm = BENCHMARK_METRICS
    mobile = bm["MobileNetV2"]
    resnet = bm["ResNet-18"]
    cnn    = bm["Custom CNN"]

    # Top transfer model — driven by the data, not hard-coded
    top = max([mobile, resnet], key=lambda m: m["test_accuracy"])
    pp_delta    = top["test_accuracy"] - cnn["test_accuracy"]
    param_ratio = round(cnn["trainable_params"] / top["trainable_params"])

    cnn_millions = cnn["trainable_params"] / 1_000_000
    return (
        f"- **ResNet-18** — {resnet['test_accuracy']:.2f}% accuracy with only "
        f"{resnet['trainable_params']:,} trainable parameters "
        f"(fewest of any deployed model).\n"
        f"- **MobileNetV2** — {mobile['test_accuracy']:.2f}% accuracy with "
        f"{mobile['trainable_params']:,} trainable parameters "
        f"(frozen backbone + linear head).\n"
        f"- **Custom CNN** — {cnn['test_accuracy']:.2f}% accuracy with "
        f"{cnn_millions:.2f}M trainable parameters "
        f"(trained from scratch — demonstrates the transfer-learning gap).\n"
        f"- **ResNet-18** achieves +{pp_delta:.1f} percentage points over the Custom CNN "
        f"with ~{param_ratio}× fewer trainable parameters."
    )


def _about_model_table_md() -> str:
    """Build the deployed-model summary table from BENCHMARK_METRICS."""
    header = (
        "| Model | Approach | Trainable Params | Accuracy |\n"
        "|-------|----------|-----------------|----------|\n"
    )
    rows = [
        f"| {m['display_name']} | {m['strategy'].split('(')[0].strip()} "
        f"| {m['trainable_params']:,} | {m['test_accuracy']:.2f}% |"
        for m in BENCHMARK_METRICS.values()
        if m["available"]
    ]
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

with gr.Blocks(title="CIFAR-10 — Pouya Alavi Naeini", css=_CSS) as demo:

    gr.Markdown(
        "<h1 class='title'>🧠 CIFAR-10 Image Classification</h1>"
        "<p class='sub'>Custom CNN · MobileNetV2 · ResNet-18 — "
        "a controlled comparison of from-scratch vs transfer learning</p>"
    )

    with gr.Tabs():

        # ── Tab 1: Live Demo ──────────────────────────────────────────────
        with gr.TabItem("🔬 Live Demo"):

            gr.Markdown(
                "> **Tip:** These models were trained on 32×32 CIFAR-10 thumbnails. "
                "They work best on simple, centred images of a single object — "
                "use the example images below for reliable results.\n"
                ">\n"
                "> On real-world high-resolution photos the Custom CNN (trained on 32×32) often *looks* "
                "more plausible because its 32×32 resize pipeline matches its training distribution, "
                "while MobileNetV2 sees a 224×224 upscale it was never trained on. "
                "This is the classic **domain gap** — not evidence that the smaller model is better."
            )

            with gr.Row():
                with gr.Column(scale=1):
                    img_in = gr.Image(type="pil", label="Upload an image", height=280)
                    # Default to MobileNetV2 if it's deployed (best accuracy);
                    # otherwise fall back to whatever the first deployed model is.
                    _default_model = "MobileNetV2" if "MobileNetV2" in DEPLOYED_MODELS else DEPLOYED_MODELS[0]
                    model_dd = gr.Dropdown(
                        choices=DEPLOYED_MODELS,
                        value=_default_model,
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
                f"#### Compare All Deployed Models\n"
                f"Load an image above, then click the button to classify it "
                f"with {', '.join(DEPLOYED_MODELS)} simultaneously."
            )
            compare_btn = gr.Button("Compare All Models ▶", variant="secondary")

            # One gr.Label per deployed model — number of columns adapts
            # automatically when DEPLOYED_MODELS changes in benchmark_data.py.
            with gr.Row():
                compare_outputs = [
                    gr.Label(num_top_classes=5, label=name)
                    for name in DEPLOYED_MODELS
                ]

            compare_btn.click(
                fn=compare_all_models,
                inputs=img_in,
                outputs=compare_outputs,
            )

        # ── Tab 2: Model Comparison ───────────────────────────────────────
        with gr.TabItem("📊 Model Comparison"):

            gr.Markdown(
                "### Architecture Benchmark\n\n"
                "All models were evaluated under a consistent training budget on the full "
                "CIFAR-10 dataset, using Adam, lr=0.001, 15 epochs, and batch size 128, "
                "with architecture-specific adaptations where required.\n\n"
                "> **Deployed in this demo:** Custom CNN, MobileNetV2, ResNet-18.  \n"
                "> EfficientNet-B0 and ViT-Small are included in the table for study comparison only."
            )
            gr.Markdown(_comparison_table_md())

            gr.Markdown("### Key Findings\n" + _key_findings_md())

            gr.Markdown(
                "### Top-5 Hardest Confusion Pairs\n\n"
                "Bidirectional misclassification counts on the full 10,000-image test set, "
                "comparing the Custom CNN (before transfer learning) to MobileNetV2 (after)."
            )
            gr.Markdown(_confusion_pairs_md())

        # ── Tab 3: About ──────────────────────────────────────────────────
        with gr.TabItem("ℹ️ About"):

            gr.Markdown(
                "## About This Project\n\n"
                "> *How much does a pretrained ImageNet backbone actually help compared to "
                "training from scratch — when architectures share the same training budget?*\n\n"
                "### Dataset\n"
                "CIFAR-10 — 60,000 **32×32** RGB images across 10 classes: "
                "airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck. "
                "50,000 training images · 10,000 test images.\n\n"
                "### ⚠️ Domain Gap — Real-World Images\n"
                "These models were trained exclusively on **32×32 thumbnail-style** images. "
                "High-resolution or complex real-world photos may produce unexpected "
                "predictions — this is the classic *domain gap*. For best results use the "
                "example images , or simple close-up shots of a single object against "
                "a plain background.\n\n"
                "### Training Setup\n"
                f"- **Optimiser:** {TRAINING_CONFIG['optimizer']} "
                f"(lr {TRAINING_CONFIG['learning_rate']}, "
                f"wd {TRAINING_CONFIG['weight_decay']}), "
                f"CosineAnnealingLR, {TRAINING_CONFIG['epochs']} epochs, "
                f"batch {TRAINING_CONFIG['batch_size']}\n"
                "- **Custom CNN:** trained from scratch with RandomCrop, HFlip, "
                "CutOut, MixUp, and CutMix augmentation\n"
                "- **Transfer models (MobileNetV2, ResNet-18):** frozen ImageNet backbone; "
                "only the classification head is trained; no augmentation\n\n"
                "### Deployed Models\n" + _about_model_table_md() + "\n\n"
                "### Links\n"
                "- [GitHub — source code]"
                "(https://github.com/mrpouyaalavi/CIFAR-10-Image-Classification)\n"
                "- [Portfolio](https://pouyaalavi.dev)\n"
                "- [LinkedIn](https://www.linkedin.com/in/pouya-alavi/)\n\n"
                "**Pouya Alavi Naeini** — BIT student, Macquarie University "
                "(AI & Web/App Development).\n\n"
                "*PyTorch · Gradio · Hugging Face Spaces*"
            )

# ── Entry point ───────────────────────────────────────────────────────────────
# HF Spaces imports this module and looks for the `demo` object.
# Calling demo.launch() here also allows `python app.py` to work locally.

if __name__ == "__main__":
    demo.launch()
