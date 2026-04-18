"""
CIFAR-10 Interactive Classification — Streamlit Demo
=====================================================

<<<<<<< HEAD
Three architectures compared on CIFAR-10:
  · Custom CNN   — trained from scratch       (48.40 %)
  · MobileNetV2  — frozen ImageNet backbone   (86.91 %)
  · ResNet-18    — frozen ImageNet backbone   (87.48 %)
=======
A web-based demo app for classifying images using the trained Custom CNN
and MobileNetV2 models. Upload any image or sample from CIFAR-10 test set.
>>>>>>> eb910c8 (Finalize HF app updates and remove tracked checkpoints)

Run:
    streamlit run app.py

<<<<<<< HEAD
Author : Pouya Alavi Naeini  (pouya@pouyaalavi.dev)
=======
Author:  Pouya Alavi
>>>>>>> eb910c8 (Finalize HF app updates and remove tracked checkpoints)
License: MIT
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from PIL import Image

<<<<<<< HEAD
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
=======
try:
    import streamlit as st
except ImportError:
    raise SystemExit(
        "Streamlit is required: pip install streamlit"
>>>>>>> eb910c8 (Finalize HF app updates and remove tracked checkpoints)
    )


<<<<<<< HEAD
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
=======
# ============================================================================
#  Constants
# ============================================================================

CLASS_NAMES = (
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
)

CIFAR_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR_STD = (0.2023, 0.1994, 0.2010)
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


# ============================================================================
#  Model Definitions
# ============================================================================

class CustomCNN(nn.Module):
    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2), nn.Dropout2d(0.25),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2), nn.Dropout2d(0.25),
            nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2), nn.Dropout2d(0.25),
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
        x = x.view(x.size(0), -1)
        return self.classifier(x)


def build_mobilenetv2(num_classes: int = 10) -> nn.Module:
    from torchvision.models import mobilenet_v2, MobileNet_V2_Weights
    model = mobilenet_v2(weights=MobileNet_V2_Weights.IMAGENET1K_V1)
    for p in model.parameters():
        p.requires_grad = False
    model.classifier = nn.Sequential(
        nn.Dropout(0.2),
        nn.Linear(model.last_channel, num_classes),
>>>>>>> eb910c8 (Finalize HF app updates and remove tracked checkpoints)
    )
    return model


# ============================================================================
#  Model Loading
# ============================================================================

def _select_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


@st.cache_resource
def load_models():
    """Load both models once and cache them across Streamlit reruns."""
    import os
    device = _select_device()
    loaded = {}

    search_dirs = ["checkpoints", "models", "."]

    # Custom CNN
    model_cnn = CustomCNN(num_classes=10)
    cnn_candidates = [
        "custom_cnn_best.pth", "custom_cnn_final.pth",
        "custom_cnn_model.pth", "custom_cnn.pth",
    ]
    for d in search_dirs:
        for c in cnn_candidates:
            p = os.path.join(d, c)
            if os.path.isfile(p):
                state = torch.load(p, map_location=device, weights_only=True)
                model_cnn.load_state_dict(state)
                model_cnn.to(device)
                model_cnn.eval()
                loaded["Custom CNN"] = model_cnn
                break
        if "Custom CNN" in loaded:
            break

    # MobileNetV2
    model_mn = build_mobilenetv2(num_classes=10)
    mn_candidates = [
        "mobilenetv2_best.pth", "mobilenetv2_final.pth",
        "mobilenet_model.pth", "mobilenet.pth",
    ]
    for d in search_dirs:
        for c in mn_candidates:
            p = os.path.join(d, c)
            if os.path.isfile(p):
                state = torch.load(p, map_location=device, weights_only=True)
                model_mn.load_state_dict(state)
                model_mn.to(device)
                model_mn.eval()
                loaded["MobileNetV2"] = model_mn
                break
        if "MobileNetV2" in loaded:
            break

    return loaded, device


# ============================================================================
#  Prediction
# ============================================================================

def predict(model, image: Image.Image, model_name: str, device, top_k: int = 5):
    """Run inference and return top-k predictions."""
    if model_name == "Custom CNN":
        transform = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
        ])
    else:
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])

<<<<<<< HEAD
with gr.Blocks(title="CIFAR-10 — Pouya Alavi Naeini") as demo:
=======
    tensor = transform(image).unsqueeze(0).to(device)
    model.eval()
    with torch.no_grad():
        logits = model(tensor)
        probs = F.softmax(logits, dim=1)[0]
>>>>>>> eb910c8 (Finalize HF app updates and remove tracked checkpoints)

    top_probs, top_indices = probs.topk(top_k)
    return [(CLASS_NAMES[i.item()], p.item() * 100) for p, i in zip(top_probs, top_indices)]


# ============================================================================
#  Streamlit UI
# ============================================================================

def main():
    st.set_page_config(page_title="CIFAR-10 Classifier", layout="wide")

    st.title("CIFAR-10 Image Classification")
    st.markdown(
        "Compare a **Custom CNN** (trained from scratch) against **MobileNetV2** "
        "(transfer learning) on image classification."
    )

    # Load models
    with st.spinner("Loading models..."):
        loaded_models, device = load_models()

    if not loaded_models:
        st.error(
            "No model checkpoints found. Run the notebook first to generate "
            "model weights in the checkpoints/ directory."
        )
        return

<<<<<<< HEAD
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
=======
    st.sidebar.header("Settings")
    available_models = list(loaded_models.keys())
    selected_models = st.sidebar.multiselect(
        "Models to compare", available_models, default=available_models,
    )
    top_k = st.sidebar.slider("Top-K predictions", 1, 10, 5)
>>>>>>> eb910c8 (Finalize HF app updates and remove tracked checkpoints)

    st.sidebar.markdown("---")
    st.sidebar.markdown(f"**Device:** `{device}`")
    st.sidebar.markdown(f"**Models loaded:** {len(loaded_models)}")

    # Input selection
    tab_upload, tab_cifar = st.tabs(["Upload Image", "CIFAR-10 Test Sample"])

    image = None
    true_label = None

    with tab_upload:
        uploaded = st.file_uploader(
            "Upload an image", type=["png", "jpg", "jpeg", "bmp", "tiff"],
        )
        if uploaded is not None:
            image = Image.open(uploaded).convert("RGB")

    with tab_cifar:
        col1, col2 = st.columns([1, 3])
        with col1:
            sample_idx = st.number_input("Image index (0-9999)", 0, 9999, 42)
            if st.button("Load sample"):
                dataset = torchvision.datasets.CIFAR10(
                    root="./data", train=False, download=True,
                    transform=transforms.ToTensor(),
                )
                tensor_img, label = dataset[sample_idx]
                image = transforms.ToPILImage()(tensor_img)
                true_label = CLASS_NAMES[label]
        with col2:
            if st.button("Random sample"):
                dataset = torchvision.datasets.CIFAR10(
                    root="./data", train=False, download=True,
                    transform=transforms.ToTensor(),
                )
                idx = np.random.randint(len(dataset))
                tensor_img, label = dataset[idx]
                image = transforms.ToPILImage()(tensor_img)
                true_label = CLASS_NAMES[label]
                st.info(f"Loaded random sample (index {idx})")

<<<<<<< HEAD
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
=======
    # Display and classify
    if image is not None:
        cols = st.columns(1 + len(selected_models))

        with cols[0]:
            st.image(image, caption="Input Image", use_container_width=True)
            if true_label:
                st.markdown(f"**True label:** {true_label}")

        for i, model_name in enumerate(selected_models):
            model = loaded_models[model_name]
            preds = predict(model, image, model_name, device, top_k=top_k)
>>>>>>> eb910c8 (Finalize HF app updates and remove tracked checkpoints)

            with cols[i + 1]:
                st.subheader(model_name)
                for cls, conf in preds:
                    st.progress(conf / 100, text=f"{cls}: {conf:.1f}%")

<<<<<<< HEAD
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
                "example images below, or simple close-up shots of a single object against "
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
                "- [Live Demo — Hugging Face Spaces]"
                "(https://mrpouyaalavi-cifar-10-image-classification.hf.space)\n"
                "- [Portfolio](https://pouyaalavi.dev)\n\n"
                "**Pouya Alavi Naeini** — BIT student, Macquarie University "
                "(AI & Web/App Development).\n\n"
                "*PyTorch · Gradio · Hugging Face Spaces*"
            )

# ── Entry point ───────────────────────────────────────────────────────────────
# HF Spaces imports this module and looks for the `demo` object.
# Calling demo.launch() here also allows `python app.py` to work locally.

if __name__ == "__main__":
    demo.launch(css=_CSS)
=======

if __name__ == "__main__":
    main()
>>>>>>> eb910c8 (Finalize HF app updates and remove tracked checkpoints)
