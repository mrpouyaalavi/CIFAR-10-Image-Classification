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
#  Page Config
# ============================================================================

st.set_page_config(
    page_title="CIFAR-10 Classifier — Pouya Alavi",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================================
#  Custom CSS — Matches pouyaalavi.dev design system
# ============================================================================
#
#  Design tokens from the portfolio:
#    --bg:           #0a0a0f   (near-black)
#    --surface:      #12121a   (card background)
#    --surface-2:    #1a1a26   (elevated surface)
#    --accent:       #7c3aed   (vivid purple)
#    --accent-light: #a78bfa   (lighter purple)
#    --secondary:    #38bdf8   (sky blue)
#    --success:      #34d399   (emerald)
#    --text:         #e8e8ed   (soft white)
#    --text-muted:   #94a3b8   (slate)
#    --border:       rgba(255,255,255,0.1)
#    --font:         'Geist', 'Inter', system-ui, sans-serif

st.markdown("""
<style>
    /* ── Import Geist-like font (Inter is the closest Google Font) ── */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    /* ── Global overrides ── */
    html, body, [class*="stApp"] {
        font-family: 'Inter', system-ui, -apple-system, sans-serif !important;
    }

    /* Remove default Streamlit padding */
    .block-container {
        padding-top: 2rem !important;
        padding-bottom: 2rem !important;
        max-width: 1100px !important;
    }

    /* ── Header area ── */
    .app-header {
        text-align: center;
        padding: 2rem 0 1.5rem 0;
    }
    .app-header h1 {
        font-size: 2.5rem;
        font-weight: 700;
        letter-spacing: -0.02em;
        background: linear-gradient(135deg, #7c3aed 0%, #38bdf8 50%, #a78bfa 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin-bottom: 0.5rem;
    }
    .app-header p {
        color: #94a3b8;
        font-size: 1.05rem;
        font-weight: 400;
        line-height: 1.6;
        max-width: 640px;
        margin: 0 auto;
    }

    /* ── Glass cards ── */
    .glass-card {
        background: linear-gradient(135deg, #12121ae6, #1a1a26b3);
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 12px;
        padding: 1.4rem;
        margin-bottom: 1rem;
        backdrop-filter: blur(12px);
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
    }
    .glass-card:hover {
        border-color: rgba(124, 58, 237, 0.3);
        box-shadow: 0 8px 32px rgba(124, 58, 237, 0.08);
    }

    /* ── Prediction result card ── */
    .pred-result {
        background: linear-gradient(135deg, #12121ae6, #1a1a26b3);
        border: 1px solid rgba(124, 58, 237, 0.3);
        border-radius: 12px;
        padding: 1.5rem;
        text-align: center;
        margin-bottom: 1rem;
        box-shadow: 0 0 20px rgba(124, 58, 237, 0.1);
    }
    .pred-result.correct {
        border-color: rgba(52, 211, 153, 0.4);
        box-shadow: 0 0 20px rgba(52, 211, 153, 0.1);
    }
    .pred-result .model-badge {
        display: inline-block;
        padding: 0.2rem 0.75rem;
        border-radius: 20px;
        font-size: 0.75rem;
        font-weight: 600;
        letter-spacing: 0.03em;
        margin-bottom: 0.6rem;
    }
    .badge-cnn {
        background: rgba(124, 58, 237, 0.15);
        color: #a78bfa;
        border: 1px solid rgba(124, 58, 237, 0.3);
    }
    .badge-mn {
        background: rgba(56, 189, 248, 0.15);
        color: #38bdf8;
        border: 1px solid rgba(56, 189, 248, 0.3);
    }
    .pred-result .pred-class {
        font-size: 1.8rem;
        font-weight: 700;
        color: #e8e8ed;
        letter-spacing: -0.01em;
        margin: 0.3rem 0;
    }
    .pred-result .pred-conf {
        font-size: 0.95rem;
        color: #94a3b8;
    }

    /* ── Progress bars (top-k predictions) ── */
    .stProgress > div > div {
        background-color: rgba(124, 58, 237, 0.15) !important;
        border-radius: 8px !important;
    }
    .stProgress > div > div > div {
        background: linear-gradient(90deg, #7c3aed, #a78bfa) !important;
        border-radius: 8px !important;
    }

    /* ── Sidebar styling ── */
    section[data-testid="stSidebar"] {
        background: #0d0d14 !important;
        border-right: 1px solid rgba(255,255,255,0.06) !important;
    }
    section[data-testid="stSidebar"] .stMarkdown h2 {
        font-size: 1.1rem;
        font-weight: 600;
        color: #e8e8ed;
        letter-spacing: -0.01em;
    }

    /* ── Tab styling ── */
    .stTabs [data-baseweb="tab-list"] {
        gap: 0;
        background: rgba(255,255,255,0.03);
        border-radius: 10px;
        padding: 4px;
        border: 1px solid rgba(255,255,255,0.06);
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px;
        color: #94a3b8;
        font-weight: 500;
        font-size: 0.9rem;
        padding: 0.5rem 1.2rem;
    }
    .stTabs [aria-selected="true"] {
        background: rgba(124, 58, 237, 0.15) !important;
        color: #a78bfa !important;
    }
    .stTabs [data-baseweb="tab-highlight"] {
        background-color: transparent !important;
    }
    .stTabs [data-baseweb="tab-border"] {
        display: none;
    }

    /* ── File uploader ── */
    [data-testid="stFileUploader"] {
        border: 2px dashed rgba(124, 58, 237, 0.25) !important;
        border-radius: 12px !important;
        background: rgba(124, 58, 237, 0.03) !important;
    }

    /* ── Buttons ── */
    .stButton > button {
        background: rgba(124, 58, 237, 0.12) !important;
        color: #a78bfa !important;
        border: 1px solid rgba(124, 58, 237, 0.3) !important;
        border-radius: 10px !important;
        font-weight: 600 !important;
        font-size: 0.85rem !important;
        letter-spacing: 0.02em;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
    }
    .stButton > button:hover {
        background: rgba(124, 58, 237, 0.25) !important;
        border-color: rgba(124, 58, 237, 0.5) !important;
        box-shadow: 0 0 20px rgba(124, 58, 237, 0.15) !important;
        transform: translateY(-1px);
    }

    /* ── Info/Success/Error alerts ── */
    .stAlert [data-testid="stNotification"] {
        border-radius: 10px !important;
    }

    /* ── Image captions ── */
    [data-testid="stImage"] {
        border-radius: 10px;
        overflow: hidden;
    }

    /* ── Dividers ── */
    hr {
        border-color: rgba(255,255,255,0.06) !important;
    }

    /* ── Section labels ── */
    .section-label {
        font-size: 0.8rem;
        font-weight: 600;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        color: #94a3b8;
        margin-bottom: 0.6rem;
    }

    /* ── Footer link row ── */
    .footer-links {
        text-align: center;
        padding: 2rem 0 1rem 0;
        border-top: 1px solid rgba(255,255,255,0.06);
        margin-top: 2rem;
    }
    .footer-links a {
        color: #94a3b8;
        text-decoration: none;
        font-size: 0.85rem;
        margin: 0 0.8rem;
        transition: color 0.2s;
    }
    .footer-links a:hover {
        color: #a78bfa;
    }
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
        "[Portfolio](https://www.pouyaalavi.dev) · "
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
    card_class = "pred-result correct" if is_correct else "pred-result"
    badge = "badge-cnn" if model_name == "Custom CNN" else "badge-mn"

    st.markdown(
        f'<div class="{card_class}">'
        f'<span class="model-badge {badge}">{model_name}</span>'
        f'<div class="pred-class">{top_class.upper()}</div>'
        f'<div class="pred-conf">{top_conf:.1f}% confidence</div>'
        f'</div>',
        unsafe_allow_html=True,
    )

    if true_label:
        if is_correct:
            st.success(f"Correct! True label: **{true_label}**")
        else:
            st.error(f"Incorrect — True label: **{true_label}**")

    # Top-K bar chart
    st.markdown(f'<div class="section-label">Top-{top_k} Predictions</div>', unsafe_allow_html=True)
    for cls, conf in preds:
        st.progress(conf / 100, text=f"{cls}: {conf:.1f}%")

    # Grad-CAM
    if show_gradcam:
        st.markdown('<div class="section-label">Grad-CAM Heatmap</div>', unsafe_allow_html=True)
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
            fig.patch.set_facecolor("#0a0a0f")
            ax.set_facecolor("#0a0a0f")
            ax.imshow(heatmap, cmap="jet")
            ax.axis("off")
            ax.set_title("Activation Map", fontsize=9, color="#94a3b8", pad=8)
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
    # ── Header ──
    st.markdown(
        '<div class="app-header">'
        '<h1>CIFAR-10 Image Classification</h1>'
        '<p>Compare a Custom CNN trained from scratch against MobileNetV2 with '
        'transfer learning — with Grad-CAM visual explanations.</p>'
        '</div>',
        unsafe_allow_html=True,
    )

    # Load models
    with st.spinner("Loading models..."):
        loaded_models, device = _load_models()

    if not loaded_models:
        st.error(
            "No model checkpoints found. Place `.pth` files in the "
            "`checkpoints/` or `models/` directory, then reload the app."
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
            sample_idx = st.number_input("Image index (0–9999)", 0, 9999, 42)
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
        st.markdown(
            '<div class="glass-card" style="text-align:center; padding:3rem 1rem;">'
            '<p style="color:#94a3b8; font-size:1rem; margin:0;">'
            'Upload an image or load a CIFAR-10 sample to get started.</p>'
            '</div>',
            unsafe_allow_html=True,
        )
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

    # ── Footer ──
    st.markdown(
        '<div class="footer-links">'
        '<a href="https://www.pouyaalavi.dev" target="_blank">Portfolio</a>'
        '<a href="https://github.com/mrpouyaalavi/CIFAR-10-Image-Classification" target="_blank">GitHub</a>'
        '<a href="https://linkedin.com/in/mrpouyaalavi" target="_blank">LinkedIn</a>'
        '</div>',
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
