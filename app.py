"""
CIFAR-10 Image Classification — Streamlit Portfolio App
========================================================

A five-tab professional portfolio site that showcases the full CIFAR-10
study: architecture comparison, live inference, Grad-CAM interpretability,
per-class error analysis, and the project's design decisions.

Run:
    streamlit run app.py

Author : Pouya Alavi  (pouya@pouyaalavi.dev)
Demo   : https://cifar10.pouyaalavi.dev
Source : https://github.com/mrpouyaalavi/CIFAR-10-Image-Classification
License: MIT
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st
import torch
import torchvision
import torchvision.transforms as transforms
from PIL import Image

from benchmark_data import (
    BENCHMARK_METRICS,
    CONFUSION_PAIRS,
    CONVERGENCE_HISTORY,
    TRAINING_CONFIG,
    available_models,
    best_model_key,
)
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
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "Get Help": "https://github.com/mrpouyaalavi/CIFAR-10-Image-Classification",
        "Report a bug": "https://github.com/mrpouyaalavi/CIFAR-10-Image-Classification/issues",
        "About": (
            "CIFAR-10 Image Classification — a deep-learning portfolio project "
            "by Pouya Alavi. Built with PyTorch and Streamlit."
        ),
    },
)


# ============================================================================
#  Custom CSS — matches pouyaalavi.dev design system
# ============================================================================

CSS = """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

    html, body, [class*="stApp"] {
        font-family: 'Inter', system-ui, -apple-system, sans-serif !important;
    }
    .block-container {
        padding-top: 2rem !important;
        padding-bottom: 2rem !important;
        max-width: 1180px !important;
    }

    /* ── Hero / header ── */
    .hero {
        text-align: center;
        padding: 2rem 0 1rem 0;
    }
    .hero .tag {
        display: inline-block;
        font-size: 0.72rem;
        font-weight: 600;
        letter-spacing: 0.14em;
        text-transform: uppercase;
        color: #a78bfa;
        background: rgba(124, 58, 237, 0.12);
        padding: 0.35rem 0.9rem;
        border: 1px solid rgba(124, 58, 237, 0.3);
        border-radius: 999px;
        margin-bottom: 1rem;
    }
    .hero h1 {
        font-size: 2.75rem;
        font-weight: 800;
        letter-spacing: -0.025em;
        line-height: 1.1;
        background: linear-gradient(135deg, #7c3aed 0%, #38bdf8 50%, #a78bfa 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin: 0 0 0.8rem 0;
    }
    .hero p.sub {
        color: #94a3b8;
        font-size: 1.08rem;
        font-weight: 400;
        line-height: 1.6;
        max-width: 720px;
        margin: 0 auto 1.2rem auto;
    }
    .hero .cta-row a {
        display: inline-block;
        margin: 0 0.3rem;
        padding: 0.55rem 1.1rem;
        font-size: 0.85rem;
        font-weight: 600;
        border-radius: 10px;
        text-decoration: none;
        transition: all 0.25s cubic-bezier(0.4, 0, 0.2, 1);
    }
    .cta-primary {
        background: linear-gradient(135deg, #7c3aed, #a78bfa);
        color: #ffffff !important;
        box-shadow: 0 6px 24px rgba(124, 58, 237, 0.28);
    }
    .cta-primary:hover {
        transform: translateY(-1px);
        box-shadow: 0 10px 32px rgba(124, 58, 237, 0.4);
    }
    .cta-ghost {
        color: #a78bfa !important;
        background: rgba(124, 58, 237, 0.08);
        border: 1px solid rgba(124, 58, 237, 0.3);
    }
    .cta-ghost:hover {
        background: rgba(124, 58, 237, 0.18);
    }

    /* ── Hero stats grid ── */
    .hero-stats {
        display: grid;
        grid-template-columns: repeat(4, 1fr);
        gap: 0.8rem;
        margin: 1.8rem auto 0 auto;
        max-width: 860px;
    }
    @media (max-width: 820px) {
        .hero-stats { grid-template-columns: repeat(2, 1fr); }
    }
    .stat-card {
        background: linear-gradient(135deg, rgba(18,18,26,0.9), rgba(26,26,38,0.7));
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 12px;
        padding: 1rem 0.8rem;
        text-align: center;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
    }
    .stat-card:hover {
        border-color: rgba(124, 58, 237, 0.35);
        transform: translateY(-2px);
    }
    .stat-card .val {
        font-size: 1.55rem;
        font-weight: 700;
        color: #e8e8ed;
        letter-spacing: -0.02em;
        line-height: 1.1;
    }
    .stat-card .val.gradient {
        background: linear-gradient(135deg, #a78bfa, #38bdf8);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    .stat-card .lab {
        font-size: 0.72rem;
        font-weight: 500;
        color: #94a3b8;
        letter-spacing: 0.05em;
        text-transform: uppercase;
        margin-top: 0.3rem;
    }

    /* ── Glass cards ── */
    .glass-card {
        background: linear-gradient(135deg, rgba(18,18,26,0.92), rgba(26,26,38,0.75));
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 14px;
        padding: 1.5rem;
        margin-bottom: 1rem;
        backdrop-filter: blur(12px);
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
    }
    .glass-card:hover {
        border-color: rgba(124, 58, 237, 0.3);
        box-shadow: 0 8px 32px rgba(124, 58, 237, 0.08);
    }
    .glass-card h3 {
        font-size: 1.05rem;
        font-weight: 600;
        color: #e8e8ed;
        margin: 0 0 0.6rem 0;
        letter-spacing: -0.01em;
    }
    .glass-card p {
        color: #94a3b8;
        font-size: 0.92rem;
        line-height: 1.55;
        margin: 0;
    }

    /* ── Prediction result card ── */
    .pred-result {
        background: linear-gradient(135deg, rgba(18,18,26,0.92), rgba(26,26,38,0.75));
        border: 1px solid rgba(124, 58, 237, 0.35);
        border-radius: 14px;
        padding: 1.6rem;
        text-align: center;
        margin-bottom: 1rem;
        box-shadow: 0 0 24px rgba(124, 58, 237, 0.1);
    }
    .pred-result.correct {
        border-color: rgba(52, 211, 153, 0.4);
        box-shadow: 0 0 24px rgba(52, 211, 153, 0.12);
    }
    .pred-result .model-badge {
        display: inline-block;
        padding: 0.28rem 0.85rem;
        border-radius: 999px;
        font-size: 0.72rem;
        font-weight: 600;
        letter-spacing: 0.05em;
        text-transform: uppercase;
        margin-bottom: 0.7rem;
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
        font-size: 1.95rem;
        font-weight: 700;
        color: #e8e8ed;
        letter-spacing: -0.015em;
        margin: 0.3rem 0 0.1rem 0;
    }
    .pred-result .pred-conf {
        font-size: 0.95rem;
        color: #94a3b8;
    }

    /* ── Progress bars ── */
    .stProgress > div > div {
        background-color: rgba(124, 58, 237, 0.12) !important;
        border-radius: 8px !important;
    }
    .stProgress > div > div > div {
        background: linear-gradient(90deg, #7c3aed, #a78bfa) !important;
        border-radius: 8px !important;
    }

    /* ── Sidebar ── */
    section[data-testid="stSidebar"] {
        background: #0d0d14 !important;
        border-right: 1px solid rgba(255,255,255,0.06) !important;
    }
    section[data-testid="stSidebar"] .stMarkdown h2 {
        font-size: 1rem;
        font-weight: 600;
        color: #e8e8ed;
        letter-spacing: -0.01em;
    }

    /* ── Top-level tabs ── */
    .stTabs [data-baseweb="tab-list"] {
        gap: 0;
        background: rgba(255,255,255,0.03);
        border-radius: 12px;
        padding: 5px;
        border: 1px solid rgba(255,255,255,0.06);
        overflow-x: auto;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 9px;
        color: #94a3b8;
        font-weight: 500;
        font-size: 0.88rem;
        padding: 0.55rem 1.2rem;
        white-space: nowrap;
    }
    .stTabs [aria-selected="true"] {
        background: rgba(124, 58, 237, 0.16) !important;
        color: #c4b5fd !important;
    }
    .stTabs [data-baseweb="tab-highlight"] { background-color: transparent !important; }
    .stTabs [data-baseweb="tab-border"] { display: none; }

    /* ── File uploader ── */
    [data-testid="stFileUploader"] {
        border: 2px dashed rgba(124, 58, 237, 0.25) !important;
        border-radius: 12px !important;
        background: rgba(124, 58, 237, 0.03) !important;
    }

    /* ── Buttons ── */
    .stButton > button {
        background: rgba(124, 58, 237, 0.12) !important;
        color: #c4b5fd !important;
        border: 1px solid rgba(124, 58, 237, 0.3) !important;
        border-radius: 10px !important;
        font-weight: 600 !important;
        font-size: 0.85rem !important;
        letter-spacing: 0.02em;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
    }
    .stButton > button:hover {
        background: rgba(124, 58, 237, 0.22) !important;
        border-color: rgba(124, 58, 237, 0.5) !important;
        box-shadow: 0 0 20px rgba(124, 58, 237, 0.15) !important;
        transform: translateY(-1px);
    }

    /* ── Dataframes ── */
    [data-testid="stDataFrame"] {
        border-radius: 12px;
        overflow: hidden;
        border: 1px solid rgba(255,255,255,0.06);
    }

    /* ── Dividers ── */
    hr { border-color: rgba(255,255,255,0.06) !important; }

    /* ── Section labels ── */
    .section-label {
        font-size: 0.72rem;
        font-weight: 600;
        letter-spacing: 0.1em;
        text-transform: uppercase;
        color: #94a3b8;
        margin: 0.5rem 0 0.6rem 0;
    }
    .section-title {
        font-size: 1.5rem;
        font-weight: 700;
        color: #e8e8ed;
        letter-spacing: -0.02em;
        margin: 0.8rem 0 0.4rem 0;
    }
    .section-sub {
        color: #94a3b8;
        font-size: 0.95rem;
        margin: 0 0 1rem 0;
        line-height: 1.55;
    }

    /* ── Footer ── */
    .footer-links {
        text-align: center;
        padding: 2rem 0 1rem 0;
        border-top: 1px solid rgba(255,255,255,0.06);
        margin-top: 2.5rem;
    }
    .footer-links a {
        color: #94a3b8;
        text-decoration: none;
        font-size: 0.85rem;
        margin: 0 0.9rem;
        transition: color 0.2s;
    }
    .footer-links a:hover { color: #a78bfa; }
    .footer-note {
        color: #64748b;
        font-size: 0.75rem;
        margin-top: 0.4rem;
    }

    /* ── Tag chips ── */
    .chip {
        display: inline-block;
        padding: 0.3rem 0.7rem;
        font-size: 0.75rem;
        font-weight: 500;
        color: #c4b5fd;
        background: rgba(124, 58, 237, 0.1);
        border: 1px solid rgba(124, 58, 237, 0.25);
        border-radius: 999px;
        margin: 0.2rem 0.3rem 0.2rem 0;
    }
    .chip.blue {
        color: #7dd3fc;
        background: rgba(56, 189, 248, 0.1);
        border-color: rgba(56, 189, 248, 0.25);
    }
    .chip.green {
        color: #6ee7b7;
        background: rgba(52, 211, 153, 0.1);
        border-color: rgba(52, 211, 153, 0.25);
    }
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)


# ============================================================================
#  Cached resources
# ============================================================================

@st.cache_resource(show_spinner=False)
def _load_models():
    device = select_device()
    return load_models(device), device


@st.cache_resource(show_spinner=False)
def _load_cifar10_test():
    return torchvision.datasets.CIFAR10(
        root="./data", train=False, download=True,
        transform=transforms.ToTensor(),
    )


# ============================================================================
#  Sidebar — global controls
# ============================================================================

def render_sidebar(loaded_models: dict, device: torch.device) -> dict:
    st.sidebar.markdown("## Controls")
    st.sidebar.caption("These apply to the **Live Demo** tab.")

    loaded_names = list(loaded_models.keys())

    if loaded_names:
        selected_model = st.sidebar.selectbox(
            "Model",
            loaded_names,
            help="Choose which model to run inference with on the Live Demo tab.",
        )
        top_k = st.sidebar.slider("Top-K predictions", 1, 10, 5)
        show_gradcam = st.sidebar.checkbox(
            "Show Grad-CAM",
            value=True,
            help="Overlay a heatmap showing which image regions influenced the prediction.",
        )
        gradcam_alpha = 0.5
        if show_gradcam:
            gradcam_alpha = st.sidebar.slider(
                "Heatmap opacity", 0.1, 0.9, 0.5, 0.05,
            )
        compare_mode = False
        if len(loaded_names) > 1:
            compare_mode = st.sidebar.checkbox(
                "Compare both models",
                value=False,
                help="Run inference with both models side-by-side.",
            )
    else:
        selected_model = None
        top_k = 5
        show_gradcam = False
        gradcam_alpha = 0.5
        compare_mode = False

    st.sidebar.markdown("---")
    st.sidebar.markdown("### Runtime")
    st.sidebar.markdown(f"**Device**  `{device}`")
    st.sidebar.markdown(f"**Loaded**  {len(loaded_names)} / 2 models")
    st.sidebar.markdown(f"**Best accuracy**  `{BENCHMARK_METRICS[best_model_key()]['test_accuracy']:.2f}%`")

    st.sidebar.markdown("---")
    st.sidebar.markdown(
        "[Portfolio](https://www.pouyaalavi.dev) · "
        "[GitHub](https://github.com/mrpouyaalavi/CIFAR-10-Image-Classification) · "
        "[LinkedIn](https://www.linkedin.com/in/pouya-alavi)"
    )

    return {
        "selected_model": selected_model,
        "top_k": top_k,
        "show_gradcam": show_gradcam,
        "gradcam_alpha": gradcam_alpha,
        "compare_mode": compare_mode,
    }


# ============================================================================
#  Tab 1 — Overview
# ============================================================================

def render_overview_tab() -> None:
    best_key = best_model_key()
    best = BENCHMARK_METRICS[best_key]

    st.markdown(
        f'''
        <div class="hero">
            <span class="tag">Deep-Learning Portfolio Project</span>
            <h1>CIFAR-10 Image Classification</h1>
            <p class="sub">
                A complete deep-learning pipeline comparing five architectures on the
                CIFAR-10 dataset. Built with PyTorch and Streamlit, featuring
                <b>{best['test_accuracy']:.2f}% test accuracy</b> from
                {best['display_name']} using transfer learning from a frozen
                ImageNet backbone, plus Grad-CAM interpretability and CPU-friendly
                inference.
            </p>
            <div class="cta-row">
                <a class="cta-primary" href="#live-demo">Try the live demo ↓</a>
                <a class="cta-ghost" href="https://github.com/mrpouyaalavi/CIFAR-10-Image-Classification" target="_blank">View source on GitHub ↗</a>
            </div>
            <div class="hero-stats">
                <div class="stat-card">
                    <div class="val gradient">{best['test_accuracy']:.2f}%</div>
                    <div class="lab">Best Test Accuracy</div>
                </div>
                <div class="stat-card">
                    <div class="val">5</div>
                    <div class="lab">Architectures Trained</div>
                </div>
                <div class="stat-card">
                    <div class="val">60K</div>
                    <div class="lab">CIFAR-10 Images</div>
                </div>
                <div class="stat-card">
                    <div class="val">{best['latency_ms']:.1f} ms</div>
                    <div class="lab">Inference Latency</div>
                </div>
            </div>
        </div>
        ''',
        unsafe_allow_html=True,
    )

    st.markdown('<div class="section-title">What this project demonstrates</div>', unsafe_allow_html=True)
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(
            '<div class="glass-card">'
            '<h3>🧠 ML Engineering</h3>'
            '<p>End-to-end training pipeline with data augmentation '
            '(CutOut, MixUp, CutMix), progressive unfreezing, and '
            'cosine-annealing learning-rate scheduling — all reproducible from '
            'a single seed.</p>'
            '</div>',
            unsafe_allow_html=True,
        )
    with col2:
        st.markdown(
            '<div class="glass-card">'
            '<h3>🔍 Interpretability</h3>'
            '<p>Grad-CAM visual explanations reveal <em>where</em> each model '
            'looks when it makes a prediction, helping surface failure modes '
            'and build trust in the system.</p>'
            '</div>',
            unsafe_allow_html=True,
        )
    with col3:
        st.markdown(
            '<div class="glass-card">'
            '<h3>🚀 Deployment</h3>'
            '<p>INT8 dynamic quantisation, command-line inference, and this '
            'Streamlit web app — three delivery surfaces for the same model, '
            'all served from a single shared model-utils module.</p>'
            '</div>',
            unsafe_allow_html=True,
        )

    st.markdown('<div class="section-title" style="margin-top:1.5rem;">Tech stack</div>', unsafe_allow_html=True)
    chips_html = (
        '<div>'
        '<span class="chip">PyTorch 2.x</span>'
        '<span class="chip">torchvision</span>'
        '<span class="chip">MobileNetV2</span>'
        '<span class="chip">Grad-CAM</span>'
        '<span class="chip">INT8 Quantisation</span>'
        '<span class="chip blue">Streamlit</span>'
        '<span class="chip blue">NumPy</span>'
        '<span class="chip blue">Pillow</span>'
        '<span class="chip green">CUDA / MPS / CPU</span>'
        '<span class="chip green">CIFAR-10</span>'
        '</div>'
    )
    st.markdown(chips_html, unsafe_allow_html=True)


# ============================================================================
#  Tab 2 — Live Demo
# ============================================================================

def render_prediction(
    model,
    model_name: str,
    image: Image.Image,
    device: torch.device,
    top_k: int,
    show_gradcam: bool,
    gradcam_alpha: float,
    true_label: str | None,
) -> None:
    preds = predict(model, image, model_name, device, top_k=top_k)
    top_class, top_conf = preds[0]

    is_correct = true_label is not None and top_class == true_label
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

    if true_label is not None:
        if is_correct:
            st.success(f"Correct — true label is **{true_label}**")
        else:
            st.error(f"Incorrect — true label is **{true_label}**")

    st.markdown(f'<div class="section-label">Top-{top_k} Predictions</div>', unsafe_allow_html=True)
    for cls, conf in preds:
        st.progress(conf / 100, text=f"{cls}: {conf:.1f}%")

    if show_gradcam:
        st.markdown('<div class="section-label">Grad-CAM Heatmap</div>', unsafe_allow_html=True)
        with st.spinner("Computing Grad-CAM…"):
            overlay, heatmap, _, _ = compute_gradcam_overlay(
                model, image, model_name, device, alpha=gradcam_alpha,
            )
        col_a, col_b = st.columns(2)
        with col_a:
            st.image(overlay, caption="Overlay", width="stretch")
        with col_b:
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(figsize=(3, 3))
            fig.patch.set_facecolor("#0a0a0f")
            ax.set_facecolor("#0a0a0f")
            ax.imshow(heatmap, cmap="jet")
            ax.axis("off")
            ax.set_title("Activation Map", fontsize=9, color="#94a3b8", pad=8)
            st.pyplot(fig, width="stretch")
            plt.close(fig)


def render_live_demo_tab(loaded_models: dict, device: torch.device, settings: dict) -> None:
    st.markdown('<a id="live-demo"></a>', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Live demo</div>', unsafe_allow_html=True)
    st.markdown(
        '<p class="section-sub">Upload any image or pick one from the CIFAR-10 test set. '
        'The model runs on-device in your browser session and shows its top predictions '
        'alongside a Grad-CAM heatmap.</p>',
        unsafe_allow_html=True,
    )

    if not loaded_models:
        st.error(
            "No model checkpoints found. Place `.pth` files in the `checkpoints/` "
            "directory, then reload the app."
        )
        return

    # ── Image input ──────────────────────────────────────────────
    if "image" not in st.session_state:
        st.session_state["image"] = None
        st.session_state["true_label"] = None

    tab_upload, tab_cifar = st.tabs(["Upload image", "CIFAR-10 test sample"])

    with tab_upload:
        uploaded = st.file_uploader(
            "Drop a PNG/JPG/BMP/TIFF (max 10 MB)",
            type=["png", "jpg", "jpeg", "bmp", "tiff"],
        )
        if uploaded is not None:
            st.session_state["image"] = Image.open(uploaded).convert("RGB")
            st.session_state["true_label"] = None

    with tab_cifar:
        col1, col2, col3 = st.columns([1.1, 1, 1])
        with col1:
            sample_idx = st.number_input("Image index (0–9999)", 0, 9999, 42)
        with col2:
            if st.button("Load sample", width="stretch"):
                dataset = _load_cifar10_test()
                tensor_img, label = dataset[int(sample_idx)]
                st.session_state["image"] = transforms.ToPILImage()(tensor_img)
                st.session_state["true_label"] = CLASS_NAMES[label]
        with col3:
            if st.button("Random sample", width="stretch"):
                dataset = _load_cifar10_test()
                idx = int(np.random.randint(len(dataset)))
                tensor_img, label = dataset[idx]
                st.session_state["image"] = transforms.ToPILImage()(tensor_img)
                st.session_state["true_label"] = CLASS_NAMES[label]
                st.caption(f"Loaded random sample (index {idx})")

    image = st.session_state["image"]
    true_label = st.session_state["true_label"]

    if image is None:
        st.markdown(
            '<div class="glass-card" style="text-align:center; padding:3rem 1rem;">'
            '<p style="color:#94a3b8; font-size:1rem; margin:0;">'
            'Upload an image or load a CIFAR-10 sample to get started.</p>'
            '</div>',
            unsafe_allow_html=True,
        )
        return

    st.markdown("---")

    # ── Determine which models to run ────────────────────────────
    if settings["compare_mode"]:
        models_to_run = list(loaded_models.keys())
    else:
        models_to_run = [settings["selected_model"]]

    if len(models_to_run) == 1:
        col_img, col_pred = st.columns([1, 2])
        with col_img:
            st.image(image, caption="Input image", width="stretch")
            if true_label is not None:
                st.caption(f"True label: **{true_label}**")
        with col_pred:
            name = models_to_run[0]
            render_prediction(
                loaded_models[name], name, image, device,
                settings["top_k"], settings["show_gradcam"],
                settings["gradcam_alpha"], true_label,
            )
    else:
        st.image(image, caption="Input image", width=220)
        if true_label is not None:
            st.caption(f"True label: **{true_label}**")
        cols = st.columns(len(models_to_run))
        for col, name in zip(cols, models_to_run):
            with col:
                render_prediction(
                    loaded_models[name], name, image, device,
                    settings["top_k"], settings["show_gradcam"],
                    settings["gradcam_alpha"], true_label,
                )


# ============================================================================
#  Tab 3 — Model Comparison
# ============================================================================

def render_models_tab() -> None:
    st.markdown('<div class="section-title">Model comparison</div>', unsafe_allow_html=True)
    st.markdown(
        '<p class="section-sub">Five architectures were trained and evaluated on the '
        'full CIFAR-10 test set. Transfer learning dominates training-from-scratch at '
        'this dataset size — but at a latency cost.</p>',
        unsafe_allow_html=True,
    )

    rows = []
    for key, m in BENCHMARK_METRICS.items():
        rows.append({
            "Model": m["display_name"] + (" ⭐" if key == best_model_key() else ""),
            "Strategy": m["strategy"],
            "Test Acc (%)": m["test_accuracy"],
            "Trainable Params": f"{m['trainable_params']:,}",
            "Size (MB)": m["size_mb"],
            "Latency (ms)": m["latency_ms"],
            "Throughput (FPS)": m["fps"],
            "Input": f"{m['input_size']}×{m['input_size']}",
            "Deployed": "✓" if m["available"] else "—",
        })
    df = pd.DataFrame(rows)
    st.dataframe(
        df,
        width="stretch",
        hide_index=True,
        column_config={
            "Test Acc (%)": st.column_config.ProgressColumn(
                "Test Acc (%)",
                min_value=0,
                max_value=100,
                format="%.2f%%",
            ),
            "Latency (ms)": st.column_config.NumberColumn(format="%.2f ms"),
            "Size (MB)": st.column_config.NumberColumn(format="%.2f MB"),
        },
    )

    st.caption(
        "⭐ = highest-accuracy model on the test set. "
        "'Deployed' indicates whether the checkpoint ships with the live app "
        "(others are omitted to stay under GitHub's file-size limits)."
    )

    # ── Convergence chart ────────────────────────────────────────
    st.markdown('<div class="section-title" style="margin-top:2rem;">Convergence</div>', unsafe_allow_html=True)
    st.markdown(
        '<p class="section-sub">Validation accuracy after each epoch. MobileNetV2 '
        'reaches ~83% on its very first pass because the ImageNet backbone already '
        'encodes general visual features — the head only has to learn a 10-class '
        'linear mapping. Training from scratch climbs much more slowly.</p>',
        unsafe_allow_html=True,
    )

    chart_df = pd.DataFrame({
        "Epoch": list(range(1, len(next(iter(CONVERGENCE_HISTORY.values()))) + 1)),
        **{name: hist for name, hist in CONVERGENCE_HISTORY.items()},
    }).set_index("Epoch")

    st.line_chart(chart_df, height=320, width="stretch")


# ============================================================================
#  Tab 4 — Error Analysis
# ============================================================================

def render_analysis_tab() -> None:
    st.markdown('<div class="section-title">Per-class error analysis</div>', unsafe_allow_html=True)
    st.markdown(
        '<p class="section-sub">CIFAR-10 has five classic confusion pairs where '
        'inter-class visual similarity makes classification hard. Transfer learning '
        'reduced misclassifications in every pair by at least 59%, and up to 78% '
        'for the trickiest pair (bird ↔ deer).</p>',
        unsafe_allow_html=True,
    )

    rows = [
        {
            "Confusion Pair": p["pair"],
            "Custom CNN (errors)": p["before"],
            "MobileNetV2 (errors)": p["after"],
            "Reduction (%)": p["reduction_pct"],
        }
        for p in CONFUSION_PAIRS
    ]
    df = pd.DataFrame(rows)
    st.dataframe(
        df,
        width="stretch",
        hide_index=True,
        column_config={
            "Reduction (%)": st.column_config.ProgressColumn(
                "Reduction (%)", min_value=0, max_value=100, format="%d%%",
            ),
        },
    )

    st.markdown('<div class="section-title" style="margin-top:2rem;">Training configuration</div>', unsafe_allow_html=True)
    cfg_col1, cfg_col2 = st.columns(2)
    with cfg_col1:
        st.markdown(
            f'''
            <div class="glass-card">
                <h3>Hyperparameters</h3>
                <p>
                    <b>Optimizer:</b> {TRAINING_CONFIG["optimizer"]}<br>
                    <b>Loss:</b> {TRAINING_CONFIG["loss_function"]}<br>
                    <b>Scheduler:</b> {TRAINING_CONFIG["scheduler"]}<br>
                    <b>Learning rate:</b> {TRAINING_CONFIG["learning_rate"]}<br>
                    <b>Weight decay:</b> {TRAINING_CONFIG["weight_decay"]}<br>
                    <b>Batch size:</b> {TRAINING_CONFIG["batch_size"]}<br>
                    <b>Epochs:</b> {TRAINING_CONFIG["epochs"]}<br>
                    <b>Seed:</b> {TRAINING_CONFIG["seed"]}
                </p>
            </div>
            ''',
            unsafe_allow_html=True,
        )
    with cfg_col2:
        st.markdown(
            f'''
            <div class="glass-card">
                <h3>Dataset</h3>
                <p>
                    <b>Name:</b> {TRAINING_CONFIG["dataset"]}<br>
                    <b>Classes:</b> {TRAINING_CONFIG["num_classes"]}<br>
                    <b>Train images:</b> {TRAINING_CONFIG["train_size"]:,}<br>
                    <b>Test images:</b> {TRAINING_CONFIG["test_size"]:,}<br>
                    <b>Resolution:</b> 32×32 RGB (natively)<br>
                    <b>Train augmentations:</b> RandomCrop, HFlip, CutOut, MixUp, CutMix<br>
                </p>
            </div>
            ''',
            unsafe_allow_html=True,
        )

    st.markdown('<div class="section-title" style="margin-top:2rem;">Key findings</div>', unsafe_allow_html=True)
    st.markdown(
        '''
        <div class="glass-card">
            <h3>Transfer learning is dominant on small image datasets</h3>
            <p>MobileNetV2 with a frozen ImageNet backbone reached <b>85.88% validation
            accuracy on its very first epoch</b> and plateaued at 86.91% within three
            epochs, while the from-scratch Custom CNN took its entire 15-epoch budget
            to reach 48.40%. The inductive bias baked into a pretrained backbone is
            effectively free data.</p>
        </div>
        <div class="glass-card">
            <h3>Freeze the backbone — and freeze its BatchNorm too</h3>
            <p>Training the linear head alone means <em>all 12 810 trainable parameters
            are in one Linear layer</em>. But you also have to pin every BatchNorm2d to
            inference mode — otherwise the running statistics drift toward CIFAR-10's
            distribution and the pretrained feature calibration collapses. This single
            subtlety was the difference between 10% and 86.91% accuracy in my retrain
            experiments.</p>
        </div>
        <div class="glass-card">
            <h3>Latency &nbsp;≠&nbsp; model size</h3>
            <p>MobileNetV2 is actually <em>smaller</em> on disk than the Custom CNN
            (8.76 MB vs 9.42 MB) but is ~17× slower per image on CPU because of the
            224×224 input resolution required by the pretrained backbone. Throughput
            is the metric that matters for real-time applications, not checkpoint
            size.</p>
        </div>
        ''',
        unsafe_allow_html=True,
    )


# ============================================================================
#  Tab 5 — About
# ============================================================================

def render_about_tab() -> None:
    st.markdown('<div class="section-title">About this project</div>', unsafe_allow_html=True)
    st.markdown(
        '''
        <p class="section-sub">
            Hi! I'm <b>Pouya Alavi</b>, a Bachelor of Information Technology student at
            Macquarie University majoring in Artificial Intelligence and Web/App
            Development, graduating November 2026. This project started as a
            university assignment and grew into a full deep-learning portfolio piece
            that I use to evaluate my own ML-engineering discipline: reproducible
            training, honest benchmarks, interpretability, and a polished delivery
            surface.
        </p>
        ''',
        unsafe_allow_html=True,
    )

    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown(
            '''
            <div class="glass-card">
                <h3>Project timeline</h3>
                <p>
                    <b>Phase 1</b> — Custom CNN from scratch (4 conv blocks, CutOut aug).<br>
                    <b>Phase 2</b> — Benchmarking 5 architectures head-to-head.<br>
                    <b>Phase 3</b> — Progressive unfreezing + cosine annealing.<br>
                    <b>Phase 4</b> — Grad-CAM interpretability layer.<br>
                    <b>Phase 5</b> — INT8 quantisation + CLI inference tool.<br>
                    <b>Phase 6</b> — This Streamlit web app + custom domain.
                </p>
            </div>
            ''',
            unsafe_allow_html=True,
        )
    with col_b:
        st.markdown(
            '''
            <div class="glass-card">
                <h3>What I focused on</h3>
                <p>
                    ✓ Single-source-of-truth benchmark data<br>
                    ✓ Shared <code>model_utils</code> module (DRY across CLI + app)<br>
                    ✓ Legacy checkpoint key remapping for backwards compatibility<br>
                    ✓ Graceful device fallback (CUDA → MPS → CPU)<br>
                    ✓ Custom CSS matching my portfolio site<br>
                    ✓ End-to-end reproducibility from a single seed
                </p>
            </div>
            ''',
            unsafe_allow_html=True,
        )

    st.markdown('<div class="section-title" style="margin-top:2rem;">Get in touch</div>', unsafe_allow_html=True)
    st.markdown(
        '''
        <div class="glass-card">
            <p>
                📧 <a href="mailto:pouya@pouyaalavi.dev" style="color:#a78bfa;">pouya@pouyaalavi.dev</a><br>
                🌐 <a href="https://www.pouyaalavi.dev" target="_blank" style="color:#a78bfa;">pouyaalavi.dev</a><br>
                💼 <a href="https://www.linkedin.com/in/pouya-alavi" target="_blank" style="color:#a78bfa;">linkedin.com/in/pouya-alavi</a><br>
                🐙 <a href="https://github.com/mrpouyaalavi" target="_blank" style="color:#a78bfa;">github.com/mrpouyaalavi</a><br>
                📍 Sydney, Australia (open to relocation to the Netherlands)
            </p>
        </div>
        ''',
        unsafe_allow_html=True,
    )


# ============================================================================
#  Main
# ============================================================================

def main() -> None:
    with st.spinner("Loading models…"):
        loaded_models, device = _load_models()

    settings = render_sidebar(loaded_models, device)

    tab_overview, tab_demo, tab_models, tab_analysis, tab_about = st.tabs([
        "🏠 Overview",
        "🔬 Live Demo",
        "📊 Models",
        "🔍 Analysis",
        "👤 About",
    ])

    with tab_overview:
        render_overview_tab()
    with tab_demo:
        render_live_demo_tab(loaded_models, device, settings)
    with tab_models:
        render_models_tab()
    with tab_analysis:
        render_analysis_tab()
    with tab_about:
        render_about_tab()

    st.markdown(
        '''
        <div class="footer-links">
            <a href="https://www.pouyaalavi.dev" target="_blank">Portfolio</a>
            <a href="https://github.com/mrpouyaalavi/CIFAR-10-Image-Classification" target="_blank">GitHub</a>
            <a href="https://www.linkedin.com/in/pouya-alavi" target="_blank">LinkedIn</a>
            <a href="mailto:pouya@pouyaalavi.dev">Email</a>
            <div class="footer-note">
                Built with PyTorch &amp; Streamlit · © 2026 Pouya Alavi · MIT license
            </div>
        </div>
        ''',
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
