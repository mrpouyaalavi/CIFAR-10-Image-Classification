"""
CIFAR-10 Image Classification — Streamlit App
==============================================

Portfolio-quality demo comparing a Custom CNN (trained from scratch)
against MobileNetV2 (transfer learning) on CIFAR-10 classification,
with optional Grad-CAM visual explanations.

Run:
    streamlit run app.py

Author:  Pouya Alavi
License: MIT
"""

import numpy as np
import streamlit as st
import torch
import torchvision
import torchvision.transforms as transforms
from PIL import Image

from model_utils import (
    CLASS_NAMES,
    compute_gradcam_overlay,
    load_models,
    predict,
    select_device,
)

# ============================================================================
#  Page Config & Custom CSS
# ============================================================================

st.set_page_config(
    page_title="CIFAR-10 Classifier",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    /* Tighten top padding */
    .block-container { padding-top: 2rem; }

    /* Prediction card styling */
    .pred-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 12px;
        padding: 1.2rem;
        color: white;
        text-align: center;
        margin-bottom: 0.8rem;
    }
    .pred-card h2 { margin: 0; font-size: 1.6rem; }
    .pred-card p { margin: 0.3rem 0 0 0; font-size: 0.95rem; opacity: 0.9; }

    /* Correct prediction highlight */
    .pred-correct {
        background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
    }

    /* Model info badges */
    .model-badge {
        display: inline-block;
        padding: 0.25rem 0.7rem;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 600;
        margin-right: 0.4rem;
    }
    .badge-cnn { background: #e3f2fd; color: #1565c0; }
    .badge-mn  { background: #fce4ec; color: #c62828; }
</style>
""", unsafe_allow_html=True)


# ============================================================================
#  Model Loading (cached)
# ============================================================================

@st.cache_resource
def _load_models():
    device = select_device()
    return load_models(device), device


# ============================================================================
#  Sidebar
# ============================================================================

def render_sidebar(available_models: list[str], device: torch.device) -> dict:
    """Render sidebar controls and return settings dict."""
    st.sidebar.markdown("## Settings")

    selected_model = st.sidebar.selectbox(
        "Model",
        available_models,
        help="Choose which model to run inference with.",
    )

    top_k = st.sidebar.slider("Top-K predictions", 1, 10, 5)

    show_gradcam = st.sidebar.checkbox(
        "Show Grad-CAM",
        value=False,
        help="Overlay a heatmap showing which image regions influenced the prediction.",
    )

    gradcam_alpha = 0.5
    if show_gradcam:
        gradcam_alpha = st.sidebar.slider(
            "Heatmap opacity", 0.1, 0.9, 0.5, 0.05,
            help="Blending factor: higher = more heatmap, lower = more original image.",
        )

    compare_mode = False
    if len(available_models) > 1:
        compare_mode = st.sidebar.checkbox(
            "Compare both models",
            value=False,
            help="Run inference with both models side-by-side.",
        )

    st.sidebar.markdown("---")
    st.sidebar.markdown(f"**Device:** `{device}`")
    st.sidebar.markdown(f"**Models loaded:** {len(available_models)}")

    st.sidebar.markdown("---")
    st.sidebar.markdown(
        "Built with [PyTorch](https://pytorch.org/) & "
        "[Streamlit](https://streamlit.io/)  \n"
        "[GitHub](https://github.com/mrpouyaalavi/CIFAR-10-Image-Classification)"
    )

    return {
        "selected_model": selected_model,
        "top_k": top_k,
        "show_gradcam": show_gradcam,
        "gradcam_alpha": gradcam_alpha,
        "compare_mode": compare_mode,
    }


# ============================================================================
#  Prediction Display
# ============================================================================

def render_predictions(
    model,
    model_name: str,
    image: Image.Image,
    device: torch.device,
    top_k: int,
    show_gradcam: bool,
    gradcam_alpha: float,
    true_label: str | None = None,
):
    """Render predictions (and optional Grad-CAM) for one model."""
    preds = predict(model, image, model_name, device, top_k=top_k)
    top_class, top_conf = preds[0]

    # Prediction header card
    is_correct = true_label and top_class == true_label
    card_class = "pred-card pred-correct" if is_correct else "pred-card"
    badge = "badge-cnn" if model_name == "Custom CNN" else "badge-mn"

    st.markdown(
        f'<div class="{card_class}">'
        f'<p><span class="model-badge {badge}">{model_name}</span></p>'
        f'<h2>{top_class.upper()}</h2>'
        f'<p>{top_conf:.1f}% confidence</p>'
        f'</div>',
        unsafe_allow_html=True,
    )

    if true_label:
        if is_correct:
            st.success(f"Correct! True label: **{true_label}**")
        else:
            st.error(f"Incorrect. True label: **{true_label}**")

    # Top-K bar chart
    st.markdown(f"**Top-{top_k} Predictions**")
    for cls, conf in preds:
        st.progress(conf / 100, text=f"{cls}: {conf:.1f}%")

    # Grad-CAM
    if show_gradcam:
        st.markdown("**Grad-CAM Heatmap**")
        with st.spinner("Computing Grad-CAM..."):
            overlay, heatmap, _, _ = compute_gradcam_overlay(
                model, image, model_name, device, alpha=gradcam_alpha,
            )
        col_a, col_b = st.columns(2)
        with col_a:
            st.image(overlay, caption="Overlay", use_container_width=True)
        with col_b:
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(figsize=(3, 3))
            ax.imshow(heatmap, cmap="jet")
            ax.axis("off")
            ax.set_title("Activation Map", fontsize=9)
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)


# ============================================================================
#  CIFAR-10 Test Sample Loader
# ============================================================================

@st.cache_resource
def _load_cifar10_test():
    return torchvision.datasets.CIFAR10(
        root="./data", train=False, download=True,
        transform=transforms.ToTensor(),
    )


# ============================================================================
#  Main App
# ============================================================================

def main():
    # Header
    st.title("CIFAR-10 Image Classification")
    st.markdown(
        "Compare a **Custom CNN** (trained from scratch) against **MobileNetV2** "
        "(transfer learning) on image classification.  \n"
        "Toggle **Grad-CAM** in the sidebar to see which image regions drive the prediction."
    )

    # Load models
    with st.spinner("Loading models..."):
        loaded_models, device = _load_models()

    if not loaded_models:
        st.error(
            "No model checkpoints found. Place `.pth` files in the "
            "`checkpoints/` or `models/` directory, then reload the app."
        )
        st.info(
            "Run the training notebook first, or download pretrained weights "
            "from the GitHub Releases page."
        )
        return

    available = list(loaded_models.keys())
    settings = render_sidebar(available, device)

    # ── Image input ─────────────────────────────────────────────
    if "image" not in st.session_state:
        st.session_state["image"] = None
        st.session_state["true_label"] = None

    tab_upload, tab_cifar = st.tabs(["Upload Image", "CIFAR-10 Test Sample"])

    with tab_upload:
        uploaded = st.file_uploader(
            "Upload an image (PNG, JPG, BMP, TIFF)",
            type=["png", "jpg", "jpeg", "bmp", "tiff"],
        )
        if uploaded is not None:
            st.session_state["image"] = Image.open(uploaded).convert("RGB")
            st.session_state["true_label"] = None

    with tab_cifar:
        col1, col2 = st.columns([1, 3])
        with col1:
            sample_idx = st.number_input("Image index (0-9999)", 0, 9999, 42)
            if st.button("Load sample"):
                dataset = _load_cifar10_test()
                tensor_img, label = dataset[sample_idx]
                st.session_state["image"] = transforms.ToPILImage()(tensor_img)
                st.session_state["true_label"] = CLASS_NAMES[label]
        with col2:
            if st.button("Random sample"):
                dataset = _load_cifar10_test()
                idx = np.random.randint(len(dataset))
                tensor_img, label = dataset[idx]
                st.session_state["image"] = transforms.ToPILImage()(tensor_img)
                st.session_state["true_label"] = CLASS_NAMES[label]
                st.info(f"Loaded random sample (index {idx})")

    # ── Classify and display ────────────────────────────────────
    image = st.session_state["image"]
    true_label = st.session_state["true_label"]

    if image is None:
        st.markdown("---")
        st.info("Upload an image or load a CIFAR-10 sample to get started.")
        return

    st.markdown("---")

    # Determine which models to run
    if settings["compare_mode"]:
        models_to_run = available
    else:
        models_to_run = [settings["selected_model"]]

    # Layout: image on left, predictions on right
    if len(models_to_run) == 1:
        col_img, col_pred = st.columns([1, 2])
        with col_img:
            st.image(image, caption="Input Image", use_container_width=True)
            if true_label:
                st.caption(f"True label: **{true_label}**")
        with col_pred:
            name = models_to_run[0]
            render_predictions(
                loaded_models[name], name, image, device,
                settings["top_k"], settings["show_gradcam"],
                settings["gradcam_alpha"], true_label,
            )
    else:
        # Compare mode: image centered, two model columns below
        st.image(image, caption="Input Image", width=200)
        if true_label:
            st.caption(f"True label: **{true_label}**")

        cols = st.columns(len(models_to_run))
        for col, name in zip(cols, models_to_run):
            with col:
                render_predictions(
                    loaded_models[name], name, image, device,
                    settings["top_k"], settings["show_gradcam"],
                    settings["gradcam_alpha"], true_label,
                )


if __name__ == "__main__":
    main()
