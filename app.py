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

import json
import os

import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
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
    describe_device,
    list_available_models,
    load_model_by_name,
    predict,
    select_device,
)

# ============================================================================
#  Top-level navigation constants
# ============================================================================
#
# We use named constants (instead of bare strings) for the segmented-control
# options because the hero CTA callback (`_go_to_demo`) writes to
# st.session_state["nav"] using one of these values — if the label in one
# place ever drifted from the other, the CTA would silently stop working.
# Keeping them here, imported by both the hero button and main(), guarantees
# they always match.
NAV_OVERVIEW = "🏠 Overview"
NAV_LIVE_DEMO = "🔬 Live Demo"
NAV_MODELS = "📊 Models"
NAV_ANALYSIS = "🔍 Analysis"
NAV_ABOUT = "👤 About"
NAV_OPTIONS = [NAV_OVERVIEW, NAV_LIVE_DEMO, NAV_MODELS, NAV_ANALYSIS, NAV_ABOUT]

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
#  Theme system — system-aware light/dark with manual override
# ============================================================================
#
# The app has three theme modes stored in st.session_state["theme"]:
#
#   "auto"  — use the OS preference via @media (prefers-color-scheme)
#   "light" — force light theme regardless of OS
#   "dark"  — force dark theme regardless of OS
#
# Persistence:
#   1. Within a session → st.session_state["theme"]
#   2. Across reloads / bookmarks → URL query param ?theme=light|dark|auto
#
# Why URL query params and not localStorage: Streamlit components run inside
# sandboxed iframes that can't read the parent's localStorage without a
# bidirectional bridge. Query params are a clean, built-in Streamlit API and
# survive reloads, tab-restore, and shared URLs — which is all a recruiter
# actually needs.
#
# Implementation:
#   CSS uses `--bg`, `--text`, etc. Custom properties. We emit ONE of three
#   stylesheet bodies depending on the mode:
#     • auto   → :root has dark values, @media (prefers-color-scheme: light)
#                flips them. No JS needed.
#     • light  → forces the light values on :root, no media query.
#     • dark   → forces dark values on :root, no media query.
#   This keeps the parent DOM manipulation to zero and works reliably inside
#   Streamlit's rendering model.


_DARK_VARS = """
    --bg:             #0a0a0f;
    --bg-raised:      #12121a;
    --sidebar-bg:     #0d0d14;
    --card-bg:        linear-gradient(135deg, rgba(18,18,26,0.92), rgba(26,26,38,0.75));
    --card-bg-solid:  #13131c;
    --stat-bg:        linear-gradient(135deg, rgba(18,18,26,0.9), rgba(26,26,38,0.7));
    --border:         rgba(255,255,255,0.08);
    --border-soft:    rgba(255,255,255,0.06);
    --border-brand:   rgba(124, 58, 237, 0.35);
    --text:           #e8e8ed;
    --text-muted:     #94a3b8;
    --text-dim:       #64748b;
    --brand:          #a78bfa;
    --brand-strong:   #7c3aed;
    --brand-accent:   #38bdf8;
    --brand-text:     #c4b5fd;
    --brand-bg-soft:  rgba(124, 58, 237, 0.12);
    --brand-bg-hover: rgba(124, 58, 237, 0.22);
    --brand-bg-sel:   rgba(124, 58, 237, 0.16);
    --chip-blue-fg:   #7dd3fc;
    --chip-blue-bg:   rgba(56, 189, 248, 0.1);
    --chip-blue-bd:   rgba(56, 189, 248, 0.25);
    --chip-green-fg:  #6ee7b7;
    --chip-green-bg:  rgba(52, 211, 153, 0.1);
    --chip-green-bd:  rgba(52, 211, 153, 0.25);
    --success:        rgba(52, 211, 153, 0.4);
    --success-glow:   rgba(52, 211, 153, 0.12);
    --brand-glow:     rgba(124, 58, 237, 0.1);
    --shadow-brand:   0 6px 24px rgba(124, 58, 237, 0.28);
    --shadow-brand-h: 0 10px 32px rgba(124, 58, 237, 0.4);
    --uploader-bg:    rgba(124, 58, 237, 0.03);
    --uploader-bd:    rgba(124, 58, 237, 0.25);
"""

_LIGHT_VARS = """
    --bg:             #f7f8fb;
    --bg-raised:      #ffffff;
    --sidebar-bg:     #ffffff;
    --card-bg:        linear-gradient(135deg, #ffffff 0%, #f8fafc 100%);
    --card-bg-solid:  #ffffff;
    --stat-bg:        linear-gradient(135deg, #ffffff 0%, #f6f7fb 100%);
    --border:         rgba(15, 23, 42, 0.10);
    --border-soft:    rgba(15, 23, 42, 0.08);
    --border-brand:   rgba(124, 58, 237, 0.42);
    --text:           #0f172a;
    --text-muted:     #475569;
    --text-dim:       #94a3b8;
    --brand:          #7c3aed;
    --brand-strong:   #6d28d9;
    --brand-accent:   #0284c7;
    --brand-text:     #6d28d9;
    --brand-bg-soft:  rgba(124, 58, 237, 0.08);
    --brand-bg-hover: rgba(124, 58, 237, 0.14);
    --brand-bg-sel:   rgba(124, 58, 237, 0.12);
    --chip-blue-fg:   #0369a1;
    --chip-blue-bg:   rgba(14, 165, 233, 0.10);
    --chip-blue-bd:   rgba(14, 165, 233, 0.30);
    --chip-green-fg:  #047857;
    --chip-green-bg:  rgba(5, 150, 105, 0.10);
    --chip-green-bd:  rgba(5, 150, 105, 0.30);
    --success:        rgba(5, 150, 105, 0.45);
    --success-glow:   rgba(5, 150, 105, 0.12);
    --brand-glow:     rgba(124, 58, 237, 0.08);
    --shadow-brand:   0 6px 24px rgba(124, 58, 237, 0.20);
    --shadow-brand-h: 0 10px 32px rgba(124, 58, 237, 0.28);
    --uploader-bg:    rgba(124, 58, 237, 0.04);
    --uploader-bd:    rgba(124, 58, 237, 0.30);
"""


def _build_css(theme_mode: str) -> str:
    """Compose the final stylesheet for the given mode (auto/light/dark)."""
    if theme_mode == "light":
        root_block = f":root {{\n{_LIGHT_VARS}\n}}"
    elif theme_mode == "dark":
        root_block = f":root {{\n{_DARK_VARS}\n}}"
    else:  # "auto"
        root_block = (
            f":root {{\n{_DARK_VARS}\n}}\n"
            "@media (prefers-color-scheme: light) {\n"
            f"  :root {{\n{_LIGHT_VARS}\n  }}\n"
            "}"
        )

    return (
        "<style>\n"
        "@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');\n"
        + root_block
        + """

    html, body, [class*="stApp"] {
        font-family: 'Inter', system-ui, -apple-system, sans-serif !important;
        background: var(--bg) !important;
        color: var(--text) !important;
    }
    .stApp { background: var(--bg) !important; }
    [data-testid="stHeader"] { background: transparent !important; }

    /* Generous top padding so the sticky Streamlit header never clips our
       top-level navigation on first paint. 3.5rem gives enough safe area
       on every viewport we've tested (desktop, tablet, narrow laptop). */
    .block-container {
        padding-top: 3.5rem !important;
        padding-bottom: 2rem !important;
        max-width: 1180px !important;
    }

    /* ── Hero / header ── */
    .hero { text-align: center; padding: 1.2rem 0 1rem 0; }
    .hero .tag {
        display: inline-block;
        font-size: 0.72rem;
        font-weight: 600;
        letter-spacing: 0.14em;
        text-transform: uppercase;
        color: var(--brand);
        background: var(--brand-bg-soft);
        padding: 0.35rem 0.9rem;
        border: 1px solid var(--border-brand);
        border-radius: 999px;
        margin-bottom: 1rem;
    }
    .hero h1 {
        font-size: 2.75rem;
        font-weight: 800;
        letter-spacing: -0.025em;
        line-height: 1.1;
        background: linear-gradient(135deg, var(--brand-strong) 0%, var(--brand-accent) 50%, var(--brand) 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin: 0 0 0.8rem 0;
    }
    .hero p.sub {
        color: var(--text-muted);
        font-size: 1.08rem;
        font-weight: 400;
        line-height: 1.6;
        max-width: 720px;
        margin: 0 auto 1.2rem auto;
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
        background: var(--stat-bg);
        border: 1px solid var(--border);
        border-radius: 12px;
        padding: 1rem 0.8rem;
        text-align: center;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
    }
    .stat-card:hover {
        border-color: var(--border-brand);
        transform: translateY(-2px);
    }
    .stat-card .val {
        font-size: 1.55rem;
        font-weight: 700;
        color: var(--text);
        letter-spacing: -0.02em;
        line-height: 1.1;
    }
    .stat-card .val.gradient {
        background: linear-gradient(135deg, var(--brand), var(--brand-accent));
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    .stat-card .lab {
        font-size: 0.72rem;
        font-weight: 500;
        color: var(--text-muted);
        letter-spacing: 0.05em;
        text-transform: uppercase;
        margin-top: 0.3rem;
    }

    /* ── Glass cards ── */
    .glass-card {
        background: var(--card-bg);
        border: 1px solid var(--border);
        border-radius: 14px;
        padding: 1.5rem;
        margin-bottom: 1rem;
        backdrop-filter: blur(12px);
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
    }
    .glass-card:hover {
        border-color: var(--border-brand);
        box-shadow: 0 8px 32px var(--brand-glow);
    }
    .glass-card h3 {
        font-size: 1.05rem;
        font-weight: 600;
        color: var(--text);
        margin: 0 0 0.6rem 0;
        letter-spacing: -0.01em;
    }
    .glass-card p, .glass-card li {
        color: var(--text-muted);
        font-size: 0.92rem;
        line-height: 1.55;
        margin: 0;
    }
    .glass-card code {
        background: var(--brand-bg-soft);
        color: var(--brand-text);
        padding: 0.08rem 0.35rem;
        border-radius: 5px;
        font-size: 0.85em;
    }

    /* ── Prediction result card ── */
    .pred-result {
        background: var(--card-bg);
        border: 1px solid var(--border-brand);
        border-radius: 14px;
        padding: 1.6rem;
        text-align: center;
        margin-bottom: 1rem;
        box-shadow: 0 0 24px var(--brand-glow);
    }
    .pred-result.correct {
        border-color: var(--success);
        box-shadow: 0 0 24px var(--success-glow);
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
        background: var(--brand-bg-soft);
        color: var(--brand);
        border: 1px solid var(--border-brand);
    }
    .badge-mn {
        background: var(--chip-blue-bg);
        color: var(--chip-blue-fg);
        border: 1px solid var(--chip-blue-bd);
    }
    .pred-result .pred-class {
        font-size: 1.95rem;
        font-weight: 700;
        color: var(--text);
        letter-spacing: -0.015em;
        margin: 0.3rem 0 0.1rem 0;
    }
    .pred-result .pred-conf {
        font-size: 0.95rem;
        color: var(--text-muted);
    }

    /* ── Progress bars ── */
    .stProgress > div > div {
        background-color: var(--brand-bg-soft) !important;
        border-radius: 8px !important;
    }
    .stProgress > div > div > div {
        background: linear-gradient(90deg, var(--brand-strong), var(--brand)) !important;
        border-radius: 8px !important;
    }

    /* ── Sidebar ── */
    section[data-testid="stSidebar"] {
        background: var(--sidebar-bg) !important;
        border-right: 1px solid var(--border-soft) !important;
    }
    section[data-testid="stSidebar"] * { color: var(--text); }
    section[data-testid="stSidebar"] .stMarkdown h2,
    section[data-testid="stSidebar"] .stMarkdown h3 {
        font-size: 1rem;
        font-weight: 600;
        color: var(--text);
        letter-spacing: -0.01em;
    }
    section[data-testid="stSidebar"] a { color: var(--brand) !important; }

    /* ── Secondary (nested) tabs — e.g. Upload / CIFAR sample ── */
    .stTabs [data-baseweb="tab-list"] {
        gap: 0;
        background: var(--brand-bg-soft);
        border-radius: 12px;
        padding: 5px;
        border: 1px solid var(--border-soft);
        overflow-x: auto;
        -webkit-overflow-scrolling: touch;
        scrollbar-width: thin;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 9px;
        color: var(--text-muted);
        font-weight: 500;
        font-size: 0.88rem;
        padding: 0.55rem 1.2rem;
        white-space: nowrap;
    }
    .stTabs [aria-selected="true"] {
        background: var(--brand-bg-sel) !important;
        color: var(--brand-text) !important;
    }
    .stTabs [data-baseweb="tab-highlight"] { background-color: transparent !important; }
    .stTabs [data-baseweb="tab-border"] { display: none; }

    /* ── Top-level navigation (st.segmented_control) ──
       Responsive rules:
       • On wide viewports the pill row is centred.
       • On narrow viewports it becomes a single-row horizontal scroller so
         labels are never truncated. */
    div[data-testid="stSegmentedControl"] {
        display: flex;
        justify-content: center;
        margin: 0.3rem 0 1rem 0;
        overflow-x: auto;
        -webkit-overflow-scrolling: touch;
        scrollbar-width: thin;
        padding-bottom: 4px;
    }
    div[data-testid="stSegmentedControl"] > div {
        background: var(--brand-bg-soft);
        border-radius: 12px;
        padding: 5px;
        border: 1px solid var(--border-soft);
        gap: 0;
        flex-wrap: nowrap !important;
        min-width: min-content;
    }
    div[data-testid="stSegmentedControl"] button {
        background: transparent !important;
        border: none !important;
        border-radius: 9px !important;
        color: var(--text-muted) !important;
        font-weight: 500 !important;
        font-size: 0.88rem !important;
        padding: 0.55rem 1.1rem !important;
        white-space: nowrap !important;
        transition: all 0.2s ease !important;
    }
    div[data-testid="stSegmentedControl"] button:hover {
        color: var(--brand-text) !important;
        background: var(--brand-bg-hover) !important;
    }
    div[data-testid="stSegmentedControl"] button[aria-pressed="true"],
    div[data-testid="stSegmentedControl"] button[data-selected="true"] {
        background: var(--brand-bg-sel) !important;
        color: var(--brand-text) !important;
    }

    /* ── Hero CTA row (native st.button + st.link_button) ── */
    .hero-cta-row { margin: 0.2rem 0 0.5rem 0; }
    .hero-cta-row .stButton > button,
    .hero-cta-row .stLinkButton > a {
        padding: 0.55rem 1.1rem !important;
        font-size: 0.85rem !important;
        font-weight: 600 !important;
        border-radius: 10px !important;
        transition: all 0.25s cubic-bezier(0.4, 0, 0.2, 1) !important;
        width: 100% !important;
    }
    .hero-cta-row .stButton > button[kind="primary"] {
        background: linear-gradient(135deg, var(--brand-strong), var(--brand)) !important;
        color: #ffffff !important;
        border: none !important;
        box-shadow: var(--shadow-brand) !important;
    }
    .hero-cta-row .stButton > button[kind="primary"]:hover {
        transform: translateY(-1px);
        box-shadow: var(--shadow-brand-h) !important;
    }
    .hero-cta-row .stLinkButton > a {
        color: var(--brand) !important;
        background: var(--brand-bg-soft) !important;
        border: 1px solid var(--border-brand) !important;
        text-decoration: none !important;
    }
    .hero-cta-row .stLinkButton > a:hover {
        background: var(--brand-bg-hover) !important;
    }

    /* ── File uploader ── */
    [data-testid="stFileUploader"] {
        border: 2px dashed var(--uploader-bd) !important;
        border-radius: 12px !important;
        background: var(--uploader-bg) !important;
    }
    [data-testid="stFileUploader"] small { color: var(--text-muted) !important; }

    /* ── Buttons ── */
    .stButton > button {
        background: var(--brand-bg-soft) !important;
        color: var(--brand-text) !important;
        border: 1px solid var(--border-brand) !important;
        border-radius: 10px !important;
        font-weight: 600 !important;
        font-size: 0.85rem !important;
        letter-spacing: 0.02em;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
    }
    .stButton > button:hover {
        background: var(--brand-bg-hover) !important;
        border-color: var(--brand) !important;
        box-shadow: 0 0 20px var(--brand-glow) !important;
        transform: translateY(-1px);
    }

    /* ── Dataframes ── */
    [data-testid="stDataFrame"] {
        border-radius: 12px;
        overflow: hidden;
        border: 1px solid var(--border-soft);
    }

    /* ── Dividers ── */
    hr { border-color: var(--border-soft) !important; }

    /* ── Section labels ── */
    .section-label {
        font-size: 0.72rem;
        font-weight: 600;
        letter-spacing: 0.1em;
        text-transform: uppercase;
        color: var(--text-muted);
        margin: 0.5rem 0 0.6rem 0;
    }
    .section-title {
        font-size: 1.5rem;
        font-weight: 700;
        color: var(--text);
        letter-spacing: -0.02em;
        margin: 0.8rem 0 0.4rem 0;
    }
    .section-sub {
        color: var(--text-muted);
        font-size: 0.95rem;
        margin: 0 0 1rem 0;
        line-height: 1.55;
    }

    /* ── Preset gallery (Live Demo) ── */
    .preset-label {
        font-size: 0.78rem;
        font-weight: 600;
        color: var(--text-muted);
        letter-spacing: 0.04em;
        text-transform: uppercase;
        margin: 1rem 0 0.5rem 0;
    }

    /* ── Model-card (Analysis tab) ── */
    .model-card {
        background: var(--card-bg);
        border: 1px solid var(--border);
        border-left: 3px solid var(--brand);
        border-radius: 10px;
        padding: 1.3rem 1.5rem;
        margin-top: 1rem;
    }
    .model-card h3 {
        font-size: 1rem;
        font-weight: 600;
        color: var(--text);
        margin: 0 0 0.4rem 0;
    }
    .model-card dl { margin: 0.4rem 0 0 0; }
    .model-card dt {
        font-size: 0.72rem;
        font-weight: 600;
        color: var(--text-muted);
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-top: 0.7rem;
    }
    .model-card dd {
        margin: 0.15rem 0 0 0;
        color: var(--text-muted);
        font-size: 0.9rem;
        line-height: 1.5;
    }

    /* ── Cold-start banner ── */
    .cold-start-note {
        display: block;
        text-align: center;
        font-size: 0.78rem;
        color: var(--text-dim);
        margin: 0.4rem auto 0 auto;
        max-width: 620px;
    }

    /* ── Footer ── */
    .footer-links {
        text-align: center;
        padding: 2rem 0 1rem 0;
        border-top: 1px solid var(--border-soft);
        margin-top: 2.5rem;
    }
    .footer-links a {
        color: var(--text-muted);
        text-decoration: none;
        font-size: 0.85rem;
        margin: 0 0.9rem;
        transition: color 0.2s;
    }
    .footer-links a:hover { color: var(--brand); }
    .footer-note {
        color: var(--text-dim);
        font-size: 0.75rem;
        margin-top: 0.4rem;
    }

    /* ── Tag chips ── */
    .chip {
        display: inline-block;
        padding: 0.3rem 0.7rem;
        font-size: 0.75rem;
        font-weight: 500;
        color: var(--brand-text);
        background: var(--brand-bg-soft);
        border: 1px solid var(--border-brand);
        border-radius: 999px;
        margin: 0.2rem 0.3rem 0.2rem 0;
    }
    .chip.blue {
        color: var(--chip-blue-fg);
        background: var(--chip-blue-bg);
        border-color: var(--chip-blue-bd);
    }
    .chip.green {
        color: var(--chip-green-fg);
        background: var(--chip-green-bg);
        border-color: var(--chip-green-bd);
    }
</style>
"""
    )


def _resolve_theme_mode() -> str:
    """Read theme mode from query params (persistent) → session_state → default.

    Normalizes to one of: "auto", "light", "dark".
    """
    valid = {"auto", "light", "dark"}

    # URL query param wins on first load so the choice survives browser reloads
    # and bookmarks. We intentionally use the new-style st.query_params API;
    # this requires Streamlit >= 1.30 which our requirements.txt pins.
    url_theme = st.query_params.get("theme")
    if url_theme in valid and "theme" not in st.session_state:
        st.session_state["theme"] = url_theme

    st.session_state.setdefault("theme", "auto")
    if st.session_state["theme"] not in valid:
        st.session_state["theme"] = "auto"
    return st.session_state["theme"]


# Apply the initial theme CSS before any widgets render so there's no flash.
_current_theme = _resolve_theme_mode()
st.markdown(_build_css(_current_theme), unsafe_allow_html=True)


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
#  Sidebar — conditional rendering
# ============================================================================
#
# Information architecture note: the sidebar previously rendered its model
# controls on every tab, even though the help-text explicitly said "these
# apply to the Live Demo tab". That inconsistency was confusing, so now:
#
#   • On Live Demo    → full controls (model, top-k, Grad-CAM, compare mode)
#   • On every other  → lightweight info panel (runtime, deploy scope, links)
#
# Both modes still show the theme toggle + external links so the user can
# change appearance and navigate out from any tab.


_TOTAL_TRAINED = len(BENCHMARK_METRICS)


def _render_theme_toggle() -> None:
    """Render the system-aware theme toggle. Persists via URL query param."""
    st.sidebar.markdown("### Appearance")
    st.sidebar.caption("Dark / Light / Auto (matches your OS).")

    options = ["Auto", "Light", "Dark"]
    current = st.session_state.get("theme", "auto")
    current_label = current.capitalize()
    try:
        default_idx = options.index(current_label)
    except ValueError:
        default_idx = 0

    choice = st.sidebar.radio(
        "Theme",
        options,
        index=default_idx,
        horizontal=True,
        label_visibility="collapsed",
        key="theme_radio",
    )
    new_mode = choice.lower()
    if new_mode != current:
        st.session_state["theme"] = new_mode
        st.query_params["theme"] = new_mode
        st.rerun()


def _render_runtime_panel(loaded_models: dict, device: torch.device) -> None:
    """Shared runtime/status block used by both sidebar modes."""
    loaded_names = list(loaded_models.keys())

    # describe_device() returns a human-readable string like
    # "MPS — Apple Silicon (arm64)" or "CUDA — NVIDIA A100 (40.0 GB)". The
    # deployed Streamlit Community Cloud instance is always CPU, but the
    # *code* still prefers CUDA → MPS → CPU locally, so we surface both
    # facts clearly instead of letting a recruiter assume CPU means the
    # whole project is CPU-only.
    try:
        device_label = describe_device(device)
    except Exception:
        device_label = str(device)

    st.sidebar.markdown("### Runtime")
    st.sidebar.markdown(f"**Inference device**  `{device_label}`")
    st.sidebar.caption(
        "Live demo runs inference only. Training was performed offline with "
        "CUDA / Apple MPS acceleration where available; the deployed instance "
        "uses CPU because the host is a shared free-tier container."
    )

    deployed = sum(1 for m in BENCHMARK_METRICS.values() if m["available"])
    st.sidebar.markdown(
        f"**Models loaded**  {len(loaded_names)} / {deployed} deployed"
    )
    st.sidebar.markdown(
        f"**Trained architectures**  {_TOTAL_TRAINED} "
        f"<span style='color:var(--text-dim);font-size:0.78rem;'>(benchmarked)</span>",
        unsafe_allow_html=True,
    )
    st.sidebar.markdown(
        f"**Best accuracy**  "
        f"`{BENCHMARK_METRICS[best_model_key()]['test_accuracy']:.2f}% "
        f"({BENCHMARK_METRICS[best_model_key()]['display_name']})`"
    )


def _render_sidebar_links() -> None:
    st.sidebar.markdown("---")
    st.sidebar.markdown(
        "[Portfolio](https://www.pouyaalavi.dev) · "
        "[GitHub](https://github.com/mrpouyaalavi/CIFAR-10-Image-Classification) · "
        "[LinkedIn](https://www.linkedin.com/in/pouya-alavi)"
    )


def _recommended_model(loaded_names: list[str]) -> str:
    """Return the model we want selected by default in the live demo.

    Preference order:
      1. Best-accuracy deployed model (e.g. MobileNetV2)
      2. Any other loaded model (falls back cleanly if only one is present)

    First impressions matter: a recruiter landing on Live Demo should see the
    strongest result, not the 48%-accuracy "from scratch" baseline.
    """
    best = best_model_key()
    if best in loaded_names:
        return best
    return loaded_names[0]


def render_live_demo_sidebar(loaded_models: dict, device: torch.device) -> dict:
    """Full control set. Only shown when the Live Demo tab is active."""
    st.sidebar.markdown("## Live demo controls")
    st.sidebar.caption("Configure the model and visualisation used on this tab.")

    loaded_names = list(loaded_models.keys())

    if loaded_names:
        default = _recommended_model(loaded_names)
        # Annotate the options so the recommended model is visually obvious
        # in the dropdown itself. We keep the raw key as the return value and
        # only rewrite the label via format_func.
        def _label(name: str) -> str:
            if name == default:
                return f"{name} · recommended"
            return name

        selected_model = st.sidebar.selectbox(
            "Model",
            loaded_names,
            index=loaded_names.index(default),
            format_func=_label,
            help=(
                "Choose the model to run inference with. "
                f"{default} is the highest-accuracy deployed model "
                f"({BENCHMARK_METRICS[default]['test_accuracy']:.1f}% on the "
                "10 000-image CIFAR-10 test set)."
            ),
        )
        top_k = st.sidebar.slider(
            "Top-K predictions", 1, 10, 5,
            help="How many class probabilities to display below the top prediction.",
        )
        show_gradcam = st.sidebar.checkbox(
            "Show Grad-CAM",
            value=True,
            help="Overlay a heatmap showing which image regions drove the prediction.",
        )
        gradcam_alpha = 0.5
        if show_gradcam:
            gradcam_alpha = st.sidebar.slider(
                "Heatmap opacity", 0.1, 0.9, 0.5, 0.05,
            )
        compare_mode = False
        if len(loaded_names) > 1:
            compare_mode = st.sidebar.checkbox(
                "Compare deployed models",
                value=False,
                help="Run every loaded model on the same image, side-by-side.",
            )
    else:
        selected_model = None
        top_k = 5
        show_gradcam = False
        gradcam_alpha = 0.5
        compare_mode = False

    st.sidebar.markdown("---")
    _render_runtime_panel(loaded_models, device)
    st.sidebar.markdown("---")
    _render_theme_toggle()
    _render_sidebar_links()

    return {
        "selected_model": selected_model,
        "top_k": top_k,
        "show_gradcam": show_gradcam,
        "gradcam_alpha": gradcam_alpha,
        "compare_mode": compare_mode,
    }


def render_info_sidebar(loaded_models: dict, device: torch.device) -> dict:
    """Lightweight sidebar rendered on every non-Live-Demo tab.

    We intentionally do NOT show model selection, top-k, Grad-CAM or compare
    toggles here — they would be dead controls on the Overview / Models /
    Analysis / About tabs. Instead we surface runtime info, a short project
    summary, and a hint on where the controls actually live.
    """
    st.sidebar.markdown("## Project")
    st.sidebar.markdown(
        '<p style="color:var(--text-muted); font-size:0.88rem; line-height:1.55; margin:0.2rem 0 0.8rem 0;">'
        "A five-architecture CIFAR-10 benchmark with Grad-CAM interpretability, "
        "reproducible from a single seed."
        "</p>",
        unsafe_allow_html=True,
    )
    st.sidebar.info(
        "Model controls live on the **Live Demo** tab.",
        icon="🔬",
    )

    st.sidebar.markdown("---")
    _render_runtime_panel(loaded_models, device)
    st.sidebar.markdown("---")
    _render_theme_toggle()
    _render_sidebar_links()

    # Return default settings; these are only consumed by the Live Demo tab
    # anyway, so values here are effectively unused on other tabs.
    loaded_names = list(loaded_models.keys())
    default_model = _recommended_model(loaded_names) if loaded_names else None
    return {
        "selected_model": default_model,
        "top_k": 5,
        "show_gradcam": True,
        "gradcam_alpha": 0.5,
        "compare_mode": False,
    }


# ============================================================================
#  Tab 1 — Overview
# ============================================================================

def _go_to_demo() -> None:
    """Hero CTA callback — switch the top-level navigation to the Live Demo tab.

    We must mutate st.session_state here (inside the on_click callback) rather
    than inline after rendering the button, because Streamlit forbids writing
    to a widget's session_state key *after* the widget has been instantiated in
    the same run. Callbacks execute on the next run BEFORE widgets, so this is
    the supported pattern.
    """
    st.session_state["nav"] = NAV_LIVE_DEMO


def render_overview_tab() -> None:
    best_key = best_model_key()
    best = BENCHMARK_METRICS[best_key]
    deployed_count = sum(1 for m in BENCHMARK_METRICS.values() if m["available"])

    # Hero block — part 1: tag, headline, subhead. We close the <div class="hero">
    # wrapper at the end of part 3 so the three markdown blocks render as a
    # single visual hero even though a native Streamlit button sits in between.
    st.markdown(
        f'''
        <div class="hero">
            <span class="tag">Deep-Learning Portfolio Project</span>
            <h1>CIFAR-10 Image Classification</h1>
            <p class="sub">
                An end-to-end deep-learning study comparing {_TOTAL_TRAINED} architectures
                on CIFAR-10 — from a 4-block CNN trained from scratch to a frozen
                ImageNet backbone. Built with PyTorch and Streamlit, topping out at
                <b>{best['test_accuracy']:.2f}% test accuracy</b> via
                {best['display_name']} transfer learning, with Grad-CAM
                interpretability and a clean inference pipeline that runs on CUDA,
                Apple MPS, or CPU.
            </p>
        </div>
        ''',
        unsafe_allow_html=True,
    )

    # Hero block — part 2: CTA row rendered as REAL Streamlit widgets so the
    # "Try the live demo" button can programmatically switch the nav via
    # on_click=_go_to_demo. We wrap it in a .hero-cta-row div purely so the
    # scoped CSS in the stylesheet can restyle these two buttons without
    # affecting every other button on the page.
    st.markdown('<div class="hero-cta-row">', unsafe_allow_html=True)
    _spacer_l, _col_cta_a, _col_cta_b, _spacer_r = st.columns([1, 1, 1, 1])
    with _col_cta_a:
        st.button(
            "Try the live demo →",
            key="hero_cta_demo",
            type="primary",
            on_click=_go_to_demo,
            use_container_width=True,
        )
    with _col_cta_b:
        st.link_button(
            "View source on GitHub ↗",
            "https://github.com/mrpouyaalavi/CIFAR-10-Image-Classification",
            use_container_width=True,
        )
    st.markdown('</div>', unsafe_allow_html=True)

    # Cold-start note — the deployed Streamlit Community Cloud container
    # sleeps after a few days of inactivity; we surface that honestly rather
    # than having the recruiter wonder why the first load is slow.
    st.markdown(
        '<p class="cold-start-note">'
        "Note: the demo may take a few seconds to wake on first load while the "
        "hosting container spins up. Subsequent interactions are fast."
        "</p>",
        unsafe_allow_html=True,
    )

    # Hero block — part 3: stats grid. Note the "5 trained · 2 deployed"
    # split — it's the single clearest signal to a recruiter that the study
    # is broader than the two checkpoints actually shipped in the demo.
    st.markdown(
        f'''
        <div class="hero" style="padding-top:0.8rem;">
            <div class="hero-stats">
                <div class="stat-card">
                    <div class="val gradient">{best['test_accuracy']:.2f}%</div>
                    <div class="lab">Best Test Accuracy</div>
                </div>
                <div class="stat-card">
                    <div class="val">{_TOTAL_TRAINED} <span style="font-size:1rem;color:var(--text-muted);">/ {deployed_count}</span></div>
                    <div class="lab">Trained · Deployed</div>
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

    # ── What recruiters should notice ──────────────────────────────
    # A scan-friendly summary of the strongest engineering signals.
    # Deliberately short so a recruiter can read it in ~15 seconds and still
    # walk away with a correct mental model of what the project is.
    st.markdown(
        '<div class="section-title" style="margin-top:1.8rem;">What to notice</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<p class="section-sub">'
        "A quick scan of the strongest engineering signals in this project."
        "</p>",
        unsafe_allow_html=True,
    )
    rec_col1, rec_col2 = st.columns(2)
    with rec_col1:
        st.markdown(
            '<div class="glass-card">'
            '<h3>📐 Rigorous ML process</h3>'
            '<p>End-to-end reproducible pipeline from a single seed: '
            'data augmentation (CutOut / MixUp / CutMix), cosine-annealing '
            'LR scheduling, progressive unfreezing, and honest benchmarks '
            'measured on the full 10 000-image test set.</p>'
            '</div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            '<div class="glass-card">'
            '<h3>🔬 Interpretability built in</h3>'
            '<p>Grad-CAM heatmaps expose <em>where</em> each model looks, '
            'making failure modes visible. Backward hooks handle frozen '
            'backbones correctly — a common footgun in transfer learning.</p>'
            '</div>',
            unsafe_allow_html=True,
        )
    with rec_col2:
        st.markdown(
            '<div class="glass-card">'
            '<h3>🚚 Three delivery surfaces</h3>'
            '<p>Notebook, CLI tool, and this Streamlit web app all share a '
            'single <code>model_utils</code> module — no copy-pasted model '
            'definitions, no config drift between surfaces.</p>'
            '</div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            '<div class="glass-card">'
            '<h3>📊 Benchmarked, not just trained</h3>'
            f'<p>{_TOTAL_TRAINED} architectures evaluated head-to-head on accuracy, '
            'parameter count, size, latency, and throughput — with per-class '
            'error analysis to explain the gaps, not just report them.</p>'
            '</div>',
            unsafe_allow_html=True,
        )

    st.markdown(
        '<div class="section-title" style="margin-top:1.5rem;">What this project demonstrates</div>',
        unsafe_allow_html=True,
    )
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
            '<p>INT8 dynamic quantisation, a command-line inference tool, and '
            "this Streamlit web app — three delivery surfaces served from a "
            'single shared <code>model_utils</code> module.</p>'
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
    # We visually differentiate MobileNetV2 (sky-blue badge) from every other
    # model (brand-purple badge) so compare-mode output is scannable at a
    # glance. The base `.model-badge` class provides the purple default; the
    # `.badge-mn` modifier switches to the blue variant.
    badge = "badge-mn" if model_name == "MobileNetV2" else ""

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


# Curated CIFAR-10 test-set indices chosen to show variety across the ten
# classes. Each shows up as a clickable thumbnail in the preset gallery so a
# recruiter can try the demo without hunting for their own image file. These
# indices are deterministic — the dataset's download is fixed — so every
# visitor sees the same gallery layout.
#
# We picked indices where the subject is centred and reasonably unambiguous;
# they're not cherry-picked to make the model look good (MobileNetV2 still
# gets Cat↔Dog wrong on index 3, which is the entire point of the Analysis
# tab's confusion-pair discussion).
PRESET_INDICES: list[tuple[int, str]] = [
    (25,   "airplane"),
    (2,    "ship"),
    (7,    "frog"),
    (6,    "automobile"),
    (18,   "horse"),
    (103,  "dog"),
]

MAX_UPLOAD_BYTES = 10 * 1024 * 1024  # Mirrors .streamlit/config.toml's maxUploadSize


def _set_cifar_sample(idx: int) -> None:
    """Streamlit button on_click callback — load a CIFAR-10 test image by index.

    Using a callback (rather than inlining the code after the button) lets us
    update st.session_state BEFORE the next rerun starts rendering widgets,
    which avoids the "cannot modify state after widget instantiated" error.
    """
    dataset = _load_cifar10_test()
    tensor_img, label = dataset[int(idx)]
    st.session_state["image"] = transforms.ToPILImage()(tensor_img)
    st.session_state["true_label"] = CLASS_NAMES[label]
    st.session_state["last_sample_idx"] = int(idx)


def _set_random_sample() -> None:
    dataset = _load_cifar10_test()
    idx = int(np.random.randint(len(dataset)))
    _set_cifar_sample(idx)


def render_live_demo_tab(loaded_models: dict, device: torch.device, settings: dict) -> None:
    # (Previously this function rendered an `<a id="live-demo"></a>` anchor so
    # the hero CTA could `href="#live-demo"` from the Overview tab. That never
    # worked because st.tabs hides inactive tab panels from the DOM, so the
    # anchor target didn't exist. We now drive navigation with
    # st.segmented_control + on_click callbacks, which *does* work.)
    st.markdown('<div class="section-title">Live demo</div>', unsafe_allow_html=True)
    st.markdown(
        '<p class="section-sub">Upload any image, pick a CIFAR-10 test sample, or try a '
        'preset below. The selected model runs inference in the hosting container and '
        'shows its top predictions alongside a Grad-CAM heatmap.</p>',
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
            "Drop or pick an image",
            type=["png", "jpg", "jpeg", "bmp", "tiff", "webp"],
            help=(
                "Supported formats: PNG, JPG, BMP, TIFF, WEBP. "
                "Max size: 10 MB. Best results on photos of the 10 CIFAR-10 "
                "classes: airplane, automobile, bird, cat, deer, dog, frog, "
                "horse, ship, truck. Images are resized to 32×32 (Custom CNN) "
                "or 224×224 (MobileNetV2) before inference."
            ),
        )
        if uploaded is not None:
            # Streamlit already validates the file type against the `type=`
            # list, but the MIME-vs-extension check it uses is weak, so we
            # double-check with PIL — which also catches corrupt files.
            if uploaded.size > MAX_UPLOAD_BYTES:
                st.error(
                    f"File is {uploaded.size / 1024 / 1024:.1f} MB, which exceeds "
                    "the 10 MB limit. Try a smaller image."
                )
            else:
                try:
                    img = Image.open(uploaded)
                    img.verify()            # Cheap integrity check
                    uploaded.seek(0)         # verify() leaves the stream at EOF
                    st.session_state["image"] = Image.open(uploaded).convert("RGB")
                    st.session_state["true_label"] = None
                except Exception as exc:
                    st.error(
                        f"Could not read this image. It may be corrupt or in an "
                        f"unsupported sub-format. Details: `{exc}`"
                    )

    with tab_cifar:
        col1, col2, col3 = st.columns([1.1, 1, 1])
        with col1:
            sample_idx = st.number_input(
                "Image index",
                min_value=0,
                max_value=9999,
                value=st.session_state.get("last_sample_idx", 42),
                help="Pick any index 0–9999 in the CIFAR-10 test set.",
            )
        with col2:
            st.button(
                "Load sample",
                width="stretch",
                on_click=_set_cifar_sample,
                args=(int(sample_idx),),
            )
        with col3:
            st.button(
                "Random sample",
                width="stretch",
                on_click=_set_random_sample,
            )
        if st.session_state.get("last_sample_idx") is not None and st.session_state.get("true_label") is not None:
            st.caption(
                f"Loaded index **{st.session_state['last_sample_idx']}** "
                f"(true label: *{st.session_state['true_label']}*)"
            )

    # ── Preset gallery — quick-try thumbnails ────────────────────
    # We lazily thumbnail the selected presets from the test set. The
    # @st.cache_resource on _load_cifar10_test() means the dataset is
    # downloaded once per session; subsequent presses just index into it.
    st.markdown('<div class="preset-label">Or try a preset</div>', unsafe_allow_html=True)
    preset_cols = st.columns(len(PRESET_INDICES))
    for col, (p_idx, p_label) in zip(preset_cols, PRESET_INDICES):
        with col:
            st.button(
                p_label,
                key=f"preset_{p_idx}",
                width="stretch",
                on_click=_set_cifar_sample,
                args=(p_idx,),
                help=f"Load CIFAR-10 test image #{p_idx} — a {p_label}.",
            )

    image = st.session_state["image"]
    true_label = st.session_state["true_label"]

    if image is None:
        st.markdown(
            '<div class="glass-card" style="text-align:center; padding:2.5rem 1.5rem; margin-top:1rem;">'
            '<h3 style="margin-bottom:0.4rem;">No image yet</h3>'
            '<p style="margin:0;">'
            'Upload a photo, pick a CIFAR-10 test index, or click a preset above to '
            'run the model. Grad-CAM and the top-K probabilities will appear here.'
            '</p>'
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
    deployed_count = sum(1 for m in BENCHMARK_METRICS.values() if m["available"])

    st.markdown('<div class="section-title">Model comparison</div>', unsafe_allow_html=True)
    st.markdown(
        f'<p class="section-sub">{_TOTAL_TRAINED} architectures were trained and '
        'evaluated head-to-head on the full 10 000-image CIFAR-10 test set. '
        f'{deployed_count} are shipped in the live demo — the rest are in the '
        'table for context but their checkpoints are not deployed '
        '(file-size and cold-start constraints on the free hosting tier). '
        'Transfer learning dominates training-from-scratch at this dataset '
        'size, but at a latency cost.</p>',
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
        "'Deployed' indicates whether the checkpoint ships with the live app. "
        "Non-deployed checkpoints were still trained and benchmarked; they are "
        "omitted from the demo for file-size and cold-start reasons on the "
        "free hosting tier, not because of missing work."
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
        '<p class="section-sub">CIFAR-10 has a handful of classic confusion pairs '
        'where inter-class visual similarity makes classification hard. Transfer '
        'learning from a frozen ImageNet backbone cut misclassifications in every '
        'pair — most dramatically on the vehicle and animal silhouettes — while '
        'Cat ↔ Dog stayed the stubbornest residual.</p>',
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
            (8.76 MB vs 9.42 MB) but is ~12× slower per image on CPU because of the
            224×224 input resolution required by the pretrained backbone. Throughput
            is the metric that matters for real-time applications, not checkpoint
            size.</p>
        </div>
        ''',
        unsafe_allow_html=True,
    )

    # ── Model card / limitations ───────────────────────────────────
    # Compact, trustworthy block that spells out what the model is and isn't.
    # Placement here (at the bottom of Analysis) keeps the landing pages
    # focused on engineering signals while still making the disclaimer easy
    # for anyone who is evaluating suitability for a real task.
    st.markdown(
        '<div class="section-title" style="margin-top:2rem;">Model card</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<p class="section-sub">A short, honest description of what the model is, '
        'what it is good for, and where its limits are.</p>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '''
        <div class="model-card">
            <h3>CIFAR-10 classifier — portfolio study</h3>
            <dl>
                <dt>Training data</dt>
                <dd>CIFAR-10: 50 000 labelled 32×32 RGB images across 10 balanced classes
                    (airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck).
                    Reserved 10 000 images for testing.</dd>

                <dt>Intended use</dt>
                <dd>Educational exploration of transfer learning, interpretability, and
                    ML-engineering discipline on a small-image benchmark. Demonstrates an
                    end-to-end pipeline suitable for portfolio review.</dd>

                <dt>Out-of-scope uses</dt>
                <dd>Not intended for real-world image classification, content moderation,
                    safety-critical systems, or any decision-making context where a wrong
                    prediction has a meaningful cost. CIFAR-10 is a low-resolution research
                    benchmark, not a production dataset.</dd>

                <dt>Known limitations</dt>
                <dd>
                    • 32×32 training resolution — loses fine detail that real photos rely on.<br>
                    • 10 classes only — anything outside these classes gets force-mapped to the
                      nearest lookalike (e.g. bicycle → automobile, wolf → dog).<br>
                    • Frozen ImageNet features inherit whatever biases exist in ImageNet.<br>
                    • Distribution shift: performance drops sharply on images that do not look
                      like centred, cleanly-cropped, daylight CIFAR-10 samples.
                </dd>

                <dt>Evaluation</dt>
                <dd>All metrics on this site come from the full 10 000-image test set and
                    are mirrored across README, training metadata JSON, and the comparison
                    table — a single source of truth in <code>benchmark_data.py</code>.</dd>
            </dl>
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
            I'm <b>Pouya Alavi</b>, a Bachelor of Information Technology student at
            Macquarie University majoring in Artificial Intelligence and Web/App
            Development. I built this project to practice the full loop of ML
            engineering I care about most: <em>reproducible training</em>,
            <em>honest benchmarking</em>, <em>interpretability</em>, and a
            <em>polished delivery surface</em>. It started as a uni assignment
            and kept growing whenever I wanted to sharpen one of those
            disciplines in isolation.
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
                <h3>Engineering focus</h3>
                <p>
                    ✓ Single source of truth for benchmark data<br>
                    ✓ Shared <code>model_utils</code> across notebook, CLI, and app<br>
                    ✓ Legacy checkpoint key remapping for backwards compatibility<br>
                    ✓ Graceful device fallback (CUDA → MPS → CPU)<br>
                    ✓ Inference-only deployment; training stays offline<br>
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
                📍 Sydney, Australia (open to relocation if needed)
            </p>
        </div>
        ''',
        unsafe_allow_html=True,
    )


# ============================================================================
#  Main
# ============================================================================

def main() -> None:
    with st.spinner("Waking models…"):
        loaded_models, device = _load_models()

    # Initialize navigation state on first load. We do this BEFORE instantiating
    # the widget so the widget picks up the default from session_state, and so
    # callbacks (like the hero CTA) can safely mutate it.
    if "nav" not in st.session_state:
        st.session_state["nav"] = NAV_OVERVIEW

    # We need to know which tab is active *before* rendering the sidebar so
    # we can pick the right sidebar variant. The segmented_control widget
    # reads its default from st.session_state["nav"], so we can peek there
    # for the correct value even though the widget hasn't rendered yet.
    active_preview = st.session_state.get("nav") or NAV_OVERVIEW

    if active_preview == NAV_LIVE_DEMO:
        settings = render_live_demo_sidebar(loaded_models, device)
    else:
        settings = render_info_sidebar(loaded_models, device)

    # NOTE: we intentionally do NOT use st.tabs here. st.tabs is a display-only
    # container with no key=, no programmatic "set active tab" API, and its
    # inactive tab panels are removed from the DOM entirely — which means the
    # hero CTA can never switch to the Live Demo tab. st.segmented_control is
    # a real widget with session_state backing, so we get a clickable CTA and
    # deep-linkable nav for free.
    selected = st.segmented_control(
        "Navigation",
        options=NAV_OPTIONS,
        key="nav",
        label_visibility="collapsed",
    )

    # Fallback: if the user somehow deselects (segmented_control allows this
    # when no default is in session_state), treat it as Overview so we never
    # render a blank page.
    active = selected or NAV_OVERVIEW

    if active == NAV_OVERVIEW:
        render_overview_tab()
    elif active == NAV_LIVE_DEMO:
        render_live_demo_tab(loaded_models, device, settings)
    elif active == NAV_MODELS:
        render_models_tab()
    elif active == NAV_ANALYSIS:
        render_analysis_tab()
    elif active == NAV_ABOUT:
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
