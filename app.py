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

Hosting note
------------
This app is deployed on Streamlit Community Cloud's free tier, which
sleeps the container after ~7 days of inactivity and cold-starts it on
the next visit (roughly 30–60 s while PyTorch and the model weights
reload into memory). We intentionally do NOT implement keep-alive
heartbeats or external pingers to work around this — the sleep behaviour
is a hosting-tier limitation, not a code bug, and gaming it would
violate Streamlit's ToS while still not fixing the underlying cost model.
The honest user-facing message is the short "may take a few seconds to
wake" note on the Overview tab. If zero cold starts become a hard
requirement, the migration path is Fly.io / Railway / a small VPS —
see the README's hosting section for specifics.
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
#
# Favicon strategy: Streamlit's `page_icon` accepts an emoji, a local file
# path, a URL, or a PIL.Image. We ship a small SVG in `assets/favicon.svg`
# (a rounded brand-gradient square with a 3-node NN glyph) and resolve it
# relative to this file so the project keeps working from any CWD. If the
# asset is ever missing at runtime — e.g. a partial clone — we degrade
# cleanly to the original emoji instead of crashing the whole app.

_FAVICON_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets", "favicon.svg")
_page_icon = _FAVICON_PATH if os.path.isfile(_FAVICON_PATH) else "🧠"

st.set_page_config(
    page_title="CIFAR-10 Classifier — Pouya Alavi",
    page_icon=_page_icon,
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
# Persistence (in priority order, strongest wins):
#   1. URL query param ?theme=light|dark|auto  (wins on this render)
#   2. localStorage[cifar10_theme]              (restored on first visit)
#   3. Session state                            (preserved across reruns)
#   4. Default "auto"                           (OS-aware via CSS media query)
#
# The URL query param is authoritative because it's the only channel we can
# read synchronously *server-side* in Python. We then run a small
# `components.html` bridge on every render that keeps localStorage and the
# URL in sync: if the URL has a theme, we mirror it into localStorage; if
# the URL has no theme but localStorage does, we navigate the parent page to
# add ?theme=<saved>, which triggers a single rerun where Python finally
# sees the preference.
#
# This gives us three wins at once:
#   • True localStorage persistence across devices, reloads, and bookmarks.
#   • Shareable links that carry a theme in the URL still work (URL wins).
#   • Graceful degradation: if localStorage is blocked (Safari private mode,
#     storage access denied), the script silently gives up and the user
#     falls back to the URL-param + session_state path, which is exactly
#     what we had before.
#
# Implementation:
#   CSS uses `--bg`, `--text`, etc. Custom properties. We emit ONE of three
#   stylesheet bodies depending on the mode:
#     • auto   → :root has dark values, @media (prefers-color-scheme: light)
#                flips them. No JS needed for the actual colours.
#     • light  → forces the light values on :root, no media query.
#     • dark   → forces dark values on :root, no media query.
#   This keeps server-side rendering authoritative and avoids a theme flash
#   on first paint — the bridge script only handles persistence.


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
    /* Links inside the contact card use the theme-aware brand var so they
       stay contrast-safe in both light and dark modes (the previous
       hardcoded #a78bfa only read correctly on the dark palette). */
    .contact-card a {
        color: var(--brand);
        text-decoration: none;
        border-bottom: 1px solid transparent;
        transition: border-color 120ms ease;
    }
    .contact-card a:hover {
        border-bottom-color: var(--brand);
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
       IMPORTANT — testid reality check (verified against Streamlit 1.56's
       compiled ButtonGroup.*.js and src.*.js bundles, not guessed):

         st.segmented_control emits
           <div data-testid="stButtonGroup">            ← container
             <button data-testid="stBaseButton-segmented_control">        ← pill
             <button data-testid="stBaseButton-segmented_controlActive"> ← selected pill
             ...

       Earlier versions of this file targeted "stSegmentedControl", which
       does NOT exist in Streamlit 1.56 — every rule in the old block was
       a silent no-op, which is why the pills rendered with BaseWeb's
       default dark surface in light mode.

       Responsive rules:
       • On wide viewports the pill row is centred.
       • On narrow viewports it becomes a single-row horizontal scroller so
         labels are never truncated.
    */
    div[data-testid="stButtonGroup"] {
        display: flex;
        justify-content: center;
        margin: 0.3rem 0 1rem 0;
        overflow-x: auto;
        -webkit-overflow-scrolling: touch;
        scrollbar-width: thin;
        padding-bottom: 4px;
        flex-wrap: nowrap !important;
        gap: 4px;
        background: var(--brand-bg-soft) !important;
        border: 1px solid var(--border-soft);
        border-radius: 12px;
        padding: 5px;
        min-width: min-content;
        width: fit-content;
        margin-left: auto;
        margin-right: auto;
    }
    /* Every BaseButton kind that segmented_control can emit — unselected
       and selected. We match by prefix so any future casing change still
       catches. */
    div[data-testid="stButtonGroup"] button[data-testid^="stBaseButton-segmented_control"],
    div[data-testid="stButtonGroup"] button[data-testid^="stBaseButton-pills"] {
        background: transparent !important;
        background-color: transparent !important;
        color: var(--text-muted) !important;
        border: 1px solid transparent !important;
        border-radius: 9px !important;
        box-shadow: none !important;
        font-weight: 500 !important;
        font-size: 0.88rem !important;
        padding: 0.55rem 1.1rem !important;
        white-space: nowrap !important;
        transition: background 0.2s ease, color 0.2s ease !important;
    }
    /* Every descendant — clear the dark BaseWeb inner-div paint that
       only shows through when you inspect with devtools. */
    div[data-testid="stButtonGroup"] button[data-testid^="stBaseButton-segmented_control"] *,
    div[data-testid="stButtonGroup"] button[data-testid^="stBaseButton-pills"] * {
        background: transparent !important;
        background-color: transparent !important;
        color: inherit !important;
    }
    div[data-testid="stButtonGroup"] button[data-testid^="stBaseButton-segmented_control"]:hover,
    div[data-testid="stButtonGroup"] button[data-testid^="stBaseButton-pills"]:hover {
        background: var(--brand-bg-hover) !important;
        color: var(--brand-text) !important;
    }
    /* Selected variant — Streamlit appends "Active" to the kind. */
    div[data-testid="stButtonGroup"] button[data-testid="stBaseButton-segmented_controlActive"],
    div[data-testid="stButtonGroup"] button[data-testid="stBaseButton-pillsActive"] {
        background: var(--brand-bg-sel) !important;
        background-color: var(--brand-bg-sel) !important;
        color: var(--brand-text) !important;
        border: 1px solid var(--border-brand) !important;
        font-weight: 600 !important;
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
    /* Testid reality check (Streamlit 1.56, verified in src.*.js):
       st.button emits
         <div data-testid="stButton">
           <button data-testid="stBaseButton-secondary">         ← the painted element
             <div>label</div>

       The painted element is the INNER <button>, keyed by kind
       (primary / secondary / tertiary / minimal / form_submit / ...).
       The outer <div data-testid="stButton"> is only a layout wrapper.
       Earlier rules of the form `.stButton > button` matched nothing
       because there's no <button> as a direct child of stButton — the
       inner <button> has its own stBaseButton-* testid. We target the
       inner button directly and also zero every descendant to kill the
       dark BaseWeb inner-div paint. */
    [data-testid="stButton"] button[data-testid^="stBaseButton-"],
    button[data-testid="stBaseButton-secondary"],
    button[data-testid="stBaseButton-tertiary"],
    button[data-testid="stBaseButton-minimal"] {
        background: var(--brand-bg-soft) !important;
        background-color: var(--brand-bg-soft) !important;
        color: var(--brand-text) !important;
        border: 1px solid var(--border-brand) !important;
        border-radius: 10px !important;
        font-weight: 600 !important;
        font-size: 0.85rem !important;
        letter-spacing: 0.02em;
        box-shadow: none !important;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
    }
    [data-testid="stButton"] button[data-testid^="stBaseButton-"] *,
    [data-testid="stButton"] button[data-testid^="stBaseButton-"] > div,
    [data-testid="stButton"] button[data-testid^="stBaseButton-"] > div > * {
        background: transparent !important;
        background-color: transparent !important;
        color: inherit !important;
    }
    [data-testid="stButton"] button[data-testid^="stBaseButton-"]:hover {
        background: var(--brand-bg-hover) !important;
        border-color: var(--brand) !important;
        box-shadow: 0 0 20px var(--brand-glow) !important;
        transform: translateY(-1px);
    }
    /* Primary variant keeps its gradient (hero CTA). */
    button[data-testid="stBaseButton-primary"] {
        background: linear-gradient(135deg, var(--brand-strong), var(--brand)) !important;
        color: #ffffff !important;
        border: none !important;
    }
    button[data-testid="stBaseButton-primary"] * {
        background: transparent !important;
        color: inherit !important;
    }

    /* ── Top-right app header chrome ──
       The "Share" / deploy / main-menu cluster lives as SIBLINGS of
       [data-testid="stToolbar"], not inside it. Streamlit gives them
       their own testids:
         stAppDeployButton    — the purple/black square next to Share
         stMainMenuButton     — the hamburger ⋮ menu
         stToolbarActionButton + stToolbarActionButtonLabel/Icon
                              — each action button inside stToolbarActions

       Earlier the rules only targeted [data-testid="stToolbar"], which
       never reached these siblings. We repaint each one here so they
       blend with the light-mode header. */
    /* Background clearing — target div wrappers and buttons only, NOT
       their inner icon elements (span / svg). Modern Streamlit renders
       toolbar icons via CSS masks (-webkit-mask-image + background-color)
       or icon fonts; blanket-clearing * { background: transparent } makes
       those icons invisible. */
    [data-testid="stAppDeployButton"],
    [data-testid="stAppDeployButton"] div,
    [data-testid="stAppDeployButton"] button,
    [data-testid="stMainMenuButton"],
    [data-testid="stMainMenuButton"] div,
    [data-testid="stMainMenuButton"] button,
    [data-testid="stToolbarActionButton"],
    [data-testid="stToolbarActionButton"] div,
    [data-testid="stToolbarActionButton"] button,
    [data-testid="stToolbarActions"],
    [data-testid="stToolbarActions"] div {
        background: transparent !important;
        background-color: transparent !important;
        box-shadow: none !important;
    }
    /* Force visible colour on EVERY descendant — handles SVGs
       (fill/stroke = currentColor), icon fonts (color), and CSS-mask
       icons (background-color inherits from currentColor when not
       cleared). */
    [data-testid="stAppDeployButton"] *,
    [data-testid="stMainMenuButton"] *,
    [data-testid="stToolbarActionButton"] *,
    [data-testid="stToolbarActions"] * {
        color: var(--text-muted) !important;
        opacity: 1 !important;
        visibility: visible !important;
    }
    [data-testid="stToolbarActions"],
    [data-testid="stToolbarActionButton"],
    [data-testid="stAppDeployButton"],
    [data-testid="stMainMenuButton"] {
        position: relative !important;
        z-index: 1000 !important;
        pointer-events: auto !important;
    }
    [data-testid="stAppDeployButton"] button,
    [data-testid="stMainMenuButton"] button,
    [data-testid="stToolbarActionButton"] button {
        color: var(--text-muted) !important;
        border: 1px solid var(--border-soft) !important;
        border-radius: 8px !important;
        opacity: 1 !important;
        visibility: visible !important;
    }
    [data-testid="stAppDeployButton"] button:hover,
    [data-testid="stMainMenuButton"] button:hover,
    [data-testid="stToolbarActionButton"] button:hover {
        background: var(--brand-bg-soft) !important;
        color: var(--brand-text) !important;
    }
    [data-testid="stAppDeployButton"] button:hover *,
    [data-testid="stMainMenuButton"] button:hover *,
    [data-testid="stToolbarActionButton"] button:hover * {
        color: var(--brand-text) !important;
    }
    [data-testid="stAppDeployButton"] svg,
    [data-testid="stMainMenuButton"] svg,
    [data-testid="stToolbarActionButton"] svg {
        color: var(--text-muted) !important;
        opacity: 1 !important;
        visibility: visible !important;
    }
    [data-testid="stAppDeployButton"] svg path,
    [data-testid="stAppDeployButton"] svg rect,
    [data-testid="stAppDeployButton"] svg circle,
    [data-testid="stAppDeployButton"] svg ellipse,
    [data-testid="stAppDeployButton"] svg polygon,
    [data-testid="stAppDeployButton"] svg polyline,
    [data-testid="stAppDeployButton"] svg line,
    [data-testid="stMainMenuButton"] svg path,
    [data-testid="stMainMenuButton"] svg rect,
    [data-testid="stMainMenuButton"] svg circle,
    [data-testid="stMainMenuButton"] svg ellipse,
    [data-testid="stMainMenuButton"] svg polygon,
    [data-testid="stMainMenuButton"] svg polyline,
    [data-testid="stMainMenuButton"] svg line,
    [data-testid="stToolbarActionButton"] svg path,
    [data-testid="stToolbarActionButton"] svg rect,
    [data-testid="stToolbarActionButton"] svg circle,
    [data-testid="stToolbarActionButton"] svg ellipse,
    [data-testid="stToolbarActionButton"] svg polygon,
    [data-testid="stToolbarActionButton"] svg polyline,
    [data-testid="stToolbarActionButton"] svg line {
        fill: currentColor !important;
        stroke: currentColor !important;
        opacity: 1 !important;
        visibility: visible !important;
    }
    [data-testid="stAppDeployButton"] svg [fill="none"],
    [data-testid="stMainMenuButton"] svg [fill="none"],
    [data-testid="stToolbarActionButton"] svg [fill="none"] {
        fill: none !important;
    }
    [data-testid="stAppDeployButton"] svg [stroke="none"],
    [data-testid="stMainMenuButton"] svg [stroke="none"],
    [data-testid="stToolbarActionButton"] svg [stroke="none"] {
        stroke: none !important;
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

    /* ============================================================
       Widget-surface theme overrides
       ============================================================
       Streamlit's config.toml pins `base = "dark"`, which means
       every BaseWeb widget we DON'T explicitly restyle falls back
       to the dark palette — i.e. a dark surface and near-white
       icons. That renders correctly in dark mode but leaks
       through in light mode as "dark boxes on a white page" and
       "invisible header icons". Every rule below uses theme-aware
       CSS vars so it produces the right result in both modes
       without duplicating the stylesheet. */

    /* Top-right toolbar: Share, Star, Edit, GitHub, and the
       hamburger menu. In light mode these SVGs were rendering
       with near-white `fill="currentColor"` against a white
       background and disappearing. Forcing `color: var(--text)`
       on every descendant repaints the icons to the theme's
       foreground colour in both modes. */
    [data-testid="stHeader"],
    [data-testid="stToolbar"],
    [data-testid="stStatusWidget"],
    [data-testid="stDecoration"] {
        background: transparent !important;
        background-color: transparent !important;
    }
    /* Clear backgrounds on div wrappers and buttons only — NOT on span
       or svg icon elements. Modern Streamlit can render toolbar icons
       via CSS masks (-webkit-mask-image + background-color) or icon
       fonts; blanket-clearing * { background: transparent } makes those
       icons invisible, leaving empty rounded boxes. */
    [data-testid="stHeader"] div,
    [data-testid="stHeader"] button,
    [data-testid="stToolbar"] div,
    [data-testid="stToolbar"] button,
    [data-testid="stStatusWidget"] div {
        background: transparent !important;
        background-color: transparent !important;
    }
    /* Paint EVERY descendant with the correct foreground colour.
       Covers SVGs (fill/stroke = currentColor), icon fonts (color),
       and CSS-mask icons (background-color follows currentColor). */
    [data-testid="stHeader"] *,
    [data-testid="stToolbar"] *,
    [data-testid="stMainMenu"] *,
    [data-testid="stDecoration"] * {
        color: var(--text-muted) !important;
        opacity: 1 !important;
        visibility: visible !important;
    }
    [data-testid="stHeader"] svg path,
    [data-testid="stHeader"] svg rect,
    [data-testid="stHeader"] svg circle,
    [data-testid="stHeader"] svg ellipse,
    [data-testid="stHeader"] svg polygon,
    [data-testid="stHeader"] svg polyline,
    [data-testid="stHeader"] svg line,
    [data-testid="stToolbar"] svg path,
    [data-testid="stToolbar"] svg rect,
    [data-testid="stToolbar"] svg circle,
    [data-testid="stToolbar"] svg ellipse,
    [data-testid="stToolbar"] svg polygon,
    [data-testid="stToolbar"] svg polyline,
    [data-testid="stToolbar"] svg line {
        fill: currentColor !important;
        stroke: currentColor !important;
        opacity: 1 !important;
        visibility: visible !important;
    }
    [data-testid="stHeader"] svg [fill="none"],
    [data-testid="stToolbar"] svg [fill="none"] {
        fill: none !important;
    }
    [data-testid="stHeader"] svg [stroke="none"],
    [data-testid="stToolbar"] svg [stroke="none"] {
        stroke: none !important;
    }
    [data-testid="stToolbar"] button:hover,
    [data-testid="stHeader"] button:hover {
        background: var(--brand-bg-soft) !important;
        background-color: var(--brand-bg-soft) !important;
        color: var(--brand) !important;
    }
    [data-testid="stToolbar"] button:hover *,
    [data-testid="stHeader"] button:hover * {
        color: var(--brand) !important;
    }

    /* Selectbox (Model picker in sidebar). BaseWeb select has a
       dark 1c1f26-ish surface by default; we repaint it to the
       card surface with theme-aware border and text. */
    [data-baseweb="select"] > div {
        background: var(--bg-raised) !important;
        border-color: var(--border) !important;
        color: var(--text) !important;
    }
    [data-baseweb="select"] input,
    [data-baseweb="select"] div[role="button"] {
        color: var(--text) !important;
    }
    [data-baseweb="select"] svg { fill: var(--text-muted) !important; }
    [data-baseweb="select"]:hover > div {
        border-color: var(--border-brand) !important;
    }
    /* Dropdown popover + menu items (opens above the viewport,
       so it's outside the normal widget tree and needs its own
       surface rules). */
    [data-baseweb="popover"] [data-baseweb="menu"],
    [data-baseweb="popover"] ul {
        background: var(--bg-raised) !important;
        border: 1px solid var(--border) !important;
        box-shadow: 0 8px 32px var(--brand-glow) !important;
    }
    [data-baseweb="popover"] li {
        background: transparent !important;
        color: var(--text) !important;
    }
    [data-baseweb="popover"] li:hover {
        background: var(--brand-bg-soft) !important;
        color: var(--brand) !important;
    }
    [data-baseweb="popover"] li[aria-selected="true"] {
        background: var(--brand-bg-sel) !important;
        color: var(--brand-text) !important;
    }

    /* Text / number inputs — the Live Demo tab uses st.number_input
       for the CIFAR sample index. BaseWeb input has the same
       dark-by-default surface problem as select. */
    [data-baseweb="input"],
    [data-baseweb="base-input"],
    [data-testid="stNumberInput"] input,
    [data-testid="stTextInput"] input,
    [data-testid="stTextArea"] textarea {
        background: var(--bg-raised) !important;
        border-color: var(--border) !important;
        color: var(--text) !important;
    }
    [data-baseweb="input"] input { color: var(--text) !important; }
    [data-baseweb="input"]:focus-within,
    [data-testid="stNumberInput"] input:focus,
    [data-testid="stTextInput"] input:focus {
        border-color: var(--brand) !important;
        box-shadow: 0 0 0 2px var(--brand-bg-soft) !important;
        outline: none !important;
    }
    /* Number input +/- step buttons */
    [data-testid="stNumberInput"] button {
        background: var(--brand-bg-soft) !important;
        color: var(--brand) !important;
        border: 1px solid var(--border) !important;
    }
    [data-testid="stNumberInput"] button:hover {
        background: var(--brand-bg-hover) !important;
    }
    [data-testid="stNumberInput"] button svg { fill: currentColor !important; }

    /* Slider (top-k, Grad-CAM opacity). BaseWeb slider uses a
       dark track colour; we repaint the *track* and *thumb*
       while keeping the brand gradient on the active fill. */
    [data-baseweb="slider"] [role="slider"] {
        background: var(--brand) !important;
        border: 2px solid var(--bg-raised) !important;
        box-shadow: 0 2px 8px var(--brand-glow) !important;
    }
    [data-testid="stSlider"] [data-baseweb="slider"] > div > div {
        background: var(--border) !important;
    }
    [data-testid="stSlider"] label,
    [data-testid="stSlider"] div[data-testid="stTickBar"] span {
        color: var(--text-muted) !important;
    }

    /* Checkbox + radio — BaseWeb renders both controls with a
       dark 1c1f26 fill when unchecked. Repaint to the raised
       surface so they sit on a white card cleanly. */
    [data-baseweb="checkbox"] span[role="checkbox"],
    [data-baseweb="checkbox"] div[role="checkbox"] {
        background: var(--bg-raised) !important;
        border-color: var(--border) !important;
    }
    [data-baseweb="checkbox"] span[aria-checked="true"],
    [data-baseweb="checkbox"] div[aria-checked="true"] {
        background: var(--brand) !important;
        border-color: var(--brand) !important;
    }
    /* BaseWeb renders the check square as an inner <span> under the
       label whose background is inlined via emotion styled-components.
       The only selector that reliably overrides it in light mode is
       the `stCheckbox` widget testid + every descendant span that
       isn't the checked state. */
    [data-testid="stCheckbox"] label > span:first-child,
    [data-testid="stCheckbox"] span[role="checkbox"],
    [data-testid="stCheckbox"] [data-baseweb="checkbox"] > div:first-child,
    [data-testid="stCheckbox"] [data-baseweb="checkbox"] span:first-child {
        background: var(--bg-raised) !important;
        background-color: var(--bg-raised) !important;
        border: 1.5px solid var(--border-brand) !important;
        border-radius: 4px !important;
    }
    [data-testid="stCheckbox"] label > span:first-child[aria-checked="true"],
    [data-testid="stCheckbox"] span[role="checkbox"][aria-checked="true"],
    [data-testid="stCheckbox"] [data-baseweb="checkbox"] span[aria-checked="true"] {
        background: var(--brand) !important;
        background-color: var(--brand) !important;
        border-color: var(--brand) !important;
    }
    [data-testid="stCheckbox"] label { color: var(--text) !important; }
    [data-baseweb="radio"] div[role="radio"] {
        background: var(--bg-raised) !important;
        border-color: var(--border) !important;
    }
    [data-baseweb="radio"] div[aria-checked="true"] {
        background: var(--brand) !important;
        border-color: var(--brand) !important;
    }
    /* Horizontal radio labels (used by the theme toggle) */
    [data-testid="stRadio"] label {
        color: var(--text) !important;
    }

    /* File uploader — we already set the outer dashed border via
       --uploader-bg / --uploader-bd. The internals (drop zone
       background, "Browse files" button, uploaded-file chip) are
       still dark in light mode and need explicit overrides. */
    [data-testid="stFileUploaderDropzone"],
    [data-testid="stFileUploaderDropzoneInstructions"] {
        background: transparent !important;
        color: var(--text-muted) !important;
    }
    [data-testid="stFileUploaderDropzoneInstructions"] span,
    [data-testid="stFileUploaderDropzoneInstructions"] small {
        color: var(--text-muted) !important;
    }
    [data-testid="stFileUploaderDropzone"] button {
        background: var(--brand-bg-soft) !important;
        color: var(--brand) !important;
        border: 1px solid var(--border-brand) !important;
    }
    [data-testid="stFileUploaderDropzone"] button:hover {
        background: var(--brand-bg-hover) !important;
    }
    [data-testid="stFileUploaderFile"] {
        background: var(--bg-raised) !important;
        border: 1px solid var(--border) !important;
        color: var(--text) !important;
    }
    [data-testid="stFileUploaderFile"] small { color: var(--text-muted) !important; }
    [data-testid="stFileUploaderFile"] svg { fill: var(--text-muted) !important; }

    /* Alert boxes (st.info / st.success / st.warning / st.error).
       Streamlit renders these through [data-testid="stAlert"]
       containers whose dark-mode background is near-opaque and
       does not read on white. Repaint them with a soft tint that
       still signals status in light mode. */
    [data-testid="stAlert"] {
        background: var(--brand-bg-soft) !important;
        border: 1px solid var(--border-brand) !important;
        color: var(--text) !important;
    }
    [data-testid="stAlert"] [data-testid="stMarkdownContainer"] {
        color: var(--text) !important;
    }
    [data-testid="stAlert"] svg { fill: var(--brand) !important; }
    /* Variant tints so success / warning / error are still
       distinguishable without redesigning them. */
    [data-testid="stAlertContentSuccess"],
    [data-testid="stNotificationContentSuccess"] {
        background: var(--chip-green-bg) !important;
        border-color: var(--chip-green-bd) !important;
    }
    [data-testid="stAlertContentSuccess"] svg,
    [data-testid="stNotificationContentSuccess"] svg { fill: var(--chip-green-fg) !important; }
    [data-testid="stAlertContentError"],
    [data-testid="stNotificationContentError"] {
        background: rgba(220, 38, 38, 0.08) !important;
        border-color: rgba(220, 38, 38, 0.30) !important;
    }
    [data-testid="stAlertContentError"] svg,
    [data-testid="stNotificationContentError"] svg { fill: #dc2626 !important; }
    [data-testid="stAlertContentWarning"],
    [data-testid="stNotificationContentWarning"] {
        background: rgba(217, 119, 6, 0.08) !important;
        border-color: rgba(217, 119, 6, 0.30) !important;
    }
    [data-testid="stAlertContentWarning"] svg,
    [data-testid="stNotificationContentWarning"] svg { fill: #d97706 !important; }

    /* Tooltips shown on `help=` hover — BaseWeb tooltip is a
       dark bubble by default. Invert it in light mode via the
       theme vars. */
    [data-baseweb="tooltip"] {
        background: var(--text) !important;
        color: var(--bg-raised) !important;
        border: 1px solid var(--border) !important;
    }

    /* Dataframe cells. The Streamlit DataFrame renders its own
       surface through a Glide canvas; we can at least repaint
       the container + header row via the public test IDs. */
    [data-testid="stDataFrame"] {
        background: var(--bg-raised) !important;
    }
    [data-testid="stDataFrame"] * { color: var(--text); }

    /* Line chart container — the convergence chart on the
       Analysis tab. Vega renders its own SVG, but we can at
       least ensure the chart wrapper has a theme-correct
       background so it doesn't float as a dark block on white. */
    [data-testid="stVegaLiteChart"],
    [data-testid="stArrowVegaLiteChart"] {
        background: var(--bg-raised) !important;
        border: 1px solid var(--border-soft) !important;
        border-radius: 12px;
        padding: 0.5rem;
    }

    /* Caption and small muted text */
    [data-testid="stCaptionContainer"],
    .stCaption {
        color: var(--text-muted) !important;
    }

    /* Inline <code> outside glass-card (e.g. runtime panel and sidebar
       chips for "Inference device" + "Best accuracy"). Streamlit's own
       base stylesheet paints `code` with a dark surface which wins on
       specificity unless we force !important here. */
    code,
    [data-testid="stMarkdownContainer"] code,
    section[data-testid="stSidebar"] code {
        background: var(--brand-bg-soft) !important;
        background-color: var(--brand-bg-soft) !important;
        color: var(--brand-text) !important;
        border-radius: 5px !important;
        padding: 0.08rem 0.35rem !important;
        border: 1px solid var(--border-soft) !important;
    }

    /* Image container — st.image wraps the <img> in a div that
       BaseWeb gives a dark fallback background to. On light-mode
       transparent PNGs (like our Grad-CAM overlays), that dark
       background bleeds through. Clear it. */
    [data-testid="stImage"] { background: transparent !important; }
    [data-testid="stImage"] figcaption,
    [data-testid="stImageCaption"] { color: var(--text-muted) !important; }

    /* Link-button secondary variant (hero CTA row) — the base
       .stLinkButton already has overrides inside .hero-cta-row;
       this is the catch-all for any other link buttons that
       might be added later. */
    .stLinkButton > a {
        color: var(--brand) !important;
        background: var(--brand-bg-soft) !important;
        border: 1px solid var(--border-brand) !important;
        text-decoration: none !important;
    }
    .stLinkButton > a:hover {
        background: var(--brand-bg-hover) !important;
    }

    /* ── Themed HTML tables ──
       We render the Models and Analysis tables as raw HTML (instead of
       st.dataframe) because Streamlit's DataFrame uses Glide Data Grid,
       which rasterises cells to a <canvas>. You cannot theme canvas
       pixels with CSS, so the Glide cells always rendered with a dark
       background in light mode. HTML tables give us full CSS control
       at the cost of losing Glide's interactive niceties (sort / resize),
       which we don't need for fixed, small reference tables. */
    .pa-table {
        width: 100%;
        border-collapse: separate;
        border-spacing: 0;
        background: var(--bg-raised);
        border: 1px solid var(--border-soft);
        border-radius: 12px;
        overflow: hidden;
        font-size: 0.88rem;
        color: var(--text);
    }
    .pa-table thead th {
        text-align: left;
        font-weight: 600;
        font-size: 0.74rem;
        letter-spacing: 0.06em;
        text-transform: uppercase;
        color: var(--text-muted);
        background: var(--brand-bg-soft);
        padding: 0.7rem 0.9rem;
        border-bottom: 1px solid var(--border-soft);
        white-space: nowrap;
    }
    .pa-table tbody td {
        padding: 0.65rem 0.9rem;
        border-bottom: 1px solid var(--border-soft);
        color: var(--text);
        vertical-align: middle;
    }
    .pa-table tbody tr:last-child td { border-bottom: none; }
    .pa-table tbody tr:hover td { background: var(--brand-bg-soft); }
    .pa-table td.num { text-align: right; font-variant-numeric: tabular-nums; }
    .pa-table td.center { text-align: center; }
    .pa-progress {
        position: relative;
        height: 18px;
        min-width: 120px;
        background: var(--brand-bg-soft);
        border: 1px solid var(--border-soft);
        border-radius: 6px;
        overflow: hidden;
    }
    .pa-progress > .fill {
        position: absolute;
        inset: 0 auto 0 0;
        background: linear-gradient(90deg, var(--brand-strong), var(--brand));
        border-radius: 6px 0 0 6px;
    }
    .pa-progress > .label {
        position: relative;
        z-index: 1;
        display: block;
        text-align: center;
        font-size: 0.78rem;
        font-weight: 600;
        line-height: 18px;
        color: var(--text);
        mix-blend-mode: normal;
    }

    /* ── Widget labels ──
       Streamlit's base="dark" theme paints widget labels near-white.
       In light mode this makes labels like "Drop or pick an image"
       invisible on the light uploader background. Override all widget
       labels to use the theme-aware text colour. */
    [data-testid="stWidgetLabel"],
    [data-testid="stWidgetLabel"] p,
    [data-testid="stWidgetLabel"] label,
    [data-testid="stWidgetLabel"] span {
        color: var(--text) !important;
    }

    /* ── Toolbar CSS-mask icon fix ──
       Some Streamlit toolbar icons render via -webkit-mask-image +
       background-color (the mask clips the bg to the icon shape).
       Our blanket "background: transparent" on header divs kills
       these icons. Re-paint masked elements inside the toolbar. */
    [data-testid="stHeader"] span[data-testid*="Icon"],
    [data-testid="stToolbar"] span[data-testid*="Icon"],
    [data-testid="stAppDeployButton"] span[data-testid*="Icon"],
    [data-testid="stMainMenuButton"] span[data-testid*="Icon"],
    [data-testid="stToolbarActionButton"] span[data-testid*="Icon"],
    [data-testid="stHeader"] span[role="img"],
    [data-testid="stToolbar"] span[role="img"] {
        background-color: var(--text-muted) !important;
    }
    /* Hide genuinely empty toolbar action wrappers — prevents blank
       rounded boxes when Streamlit reserves a slot but populates no icon. */
    [data-testid="stToolbarActions"] > div:empty {
        display: none !important;
    }
    /* NOTE — Platform limitation: the three-dot menu popup (Rerun, Clear
       cache, Print, Record screen, About) is rendered by Streamlit's
       BaseWeb layer and cannot be reliably themed from app-level CSS.
       Its appearance may not match the app theme in all modes. */
</style>
"""
    )


_VALID_THEMES = {"auto", "light", "dark"}


def _resolve_theme_mode() -> tuple[str, bool]:
    """Resolve the active theme and report whether the URL carried it.

    Returns
    -------
    (mode, url_had_theme)
        * mode           - normalised theme string from {"auto","light","dark"}
        * url_had_theme  - True iff st.query_params explicitly set ?theme=…
                           on this request. The localStorage bridge uses this
                           to decide whether to mirror URL→localStorage or
                           restore localStorage→URL.
    """
    # URL query param is the authoritative channel: it's synchronous,
    # server-side, and survives browser reloads / bookmarks. We intentionally
    # use the new-style st.query_params API (Streamlit >= 1.30).
    url_theme = st.query_params.get("theme")
    url_had_theme = url_theme in _VALID_THEMES
    if url_had_theme:
        st.session_state["theme"] = url_theme

    st.session_state.setdefault("theme", "auto")
    if st.session_state["theme"] not in _VALID_THEMES:
        st.session_state["theme"] = "auto"
    return st.session_state["theme"], url_had_theme


def _inject_theme_bridge(current_theme: str, url_had_theme: bool) -> None:
    """Run a tiny JS snippet that reconciles localStorage with the URL.

    This is the only piece of client-side JS in the app. It lives inside a
    zero-height Streamlit iframe (`components.html(..., height=0)`) and runs
    once per rerun. On each run it does exactly one of three things:

      (a) URL has ?theme=…   → persist that value into localStorage, so the
                                next visit without a query param remembers it.
      (b) URL has no ?theme=, localStorage has a saved value → navigate the
          parent page to add ?theme=<saved>. Python will then see it on the
          next rerun and honour it via `_resolve_theme_mode()`.
      (c) Neither URL nor localStorage has a value → do nothing. The server
          has already rendered in "auto" mode, which falls through to the
          OS preference via CSS `@media (prefers-color-scheme)`.

    All storage / navigation access is wrapped in try/catch so Safari
    private-mode or "Block all cookies" users degrade to URL-param-only
    persistence without any console spam.

    Security:
      We build the payload with `json.dumps` so the injected values survive
      HTML escaping and can't break out of the script context, even though
      the only values we emit are our own constants.
    """
    payload = json.dumps({
        "currentTheme": current_theme,
        "urlHadTheme": url_had_theme,
    })
    html = f"""
<script>
(function() {{
  try {{
    const data = {payload};
    const LS_KEY = "cifar10_theme";
    const VALID = ["auto", "light", "dark"];

    // Try the parent page's localStorage first (works when the component
    // iframe is same-origin with the Streamlit host). If cross-origin
    // blocks it, fall back to the iframe's own storage — the bridge still
    // works for same-browser persistence, just not across siblings.
    let store;
    try {{ store = window.parent.localStorage; }}
    catch (e) {{ store = window.localStorage; }}

    const saved = store.getItem(LS_KEY);

    if (data.urlHadTheme) {{
      // URL was authoritative this render → mirror into localStorage.
      if (saved !== data.currentTheme) {{
        try {{ store.setItem(LS_KEY, data.currentTheme); }} catch (e) {{}}
      }}
      return;
    }}

    // URL had nothing. If localStorage has a valid saved value, push it
    // onto the parent URL so Python picks it up on the next rerun.
    if (saved && VALID.indexOf(saved) !== -1) {{
      let targetHref = null;
      try {{
        const parentUrl = new URL(window.parent.location.href);
        parentUrl.searchParams.set("theme", saved);
        targetHref = parentUrl.toString();
      }} catch (e) {{
        // Cross-origin: use document.referrer as a best-effort fallback.
        try {{
          const refUrl = new URL(document.referrer);
          refUrl.searchParams.set("theme", saved);
          targetHref = refUrl.toString();
        }} catch (e2) {{
          return;
        }}
      }}
      if (targetHref) {{
        try {{ window.parent.location.href = targetHref; }}
        catch (e) {{ window.top.location.href = targetHref; }}
      }}
    }}
  }} catch (e) {{
    // Storage or navigation blocked. Degrade silently — URL-param-only
    // persistence still works.
    if (window.console) {{ console.warn("[cifar10] theme bridge disabled:", e); }}
  }}
}})();
</script>
"""
    # Prefer the modern `st.html(..., unsafe_allow_javascript=True)` API when
    # it exists (Streamlit ≥ 1.46), which silences the "components.v1.html
    # will be removed after 2026-06-01" deprecation warning and keeps this
    # app compatible with Streamlit's future releases. On older runtimes
    # we fall back to the legacy components API, which works identically.
    try:
        st.html(html, unsafe_allow_javascript=True)  # type: ignore[call-arg]
    except (AttributeError, TypeError):
        components.html(html, height=0)


# Apply the initial theme CSS before any widgets render so there's no flash.
_current_theme, _url_had_theme = _resolve_theme_mode()
st.markdown(_build_css(_current_theme), unsafe_allow_html=True)
_inject_theme_bridge(_current_theme, _url_had_theme)


# ============================================================================
#  Cached resources — lazy model loading
# ============================================================================
#
# Cold-start philosophy: on a free-tier container every megabyte and every
# second of import-time counts, so we do the absolute minimum at boot:
#
#   1. `_get_device()`       → pick CUDA/MPS/CPU, cap CPU threads, cache forever
#   2. `_available_models()` → list filenames on disk, nothing instantiated
#   3. `_load_model_cached()`→ invoked only when the user actually needs a
#                              specific model for inference. Streamlit's
#                              `cache_resource` holds it in memory for the
#                              whole process, so subsequent reruns are free.
#
# This replaces the old eager `_load_models()` that loaded every checkpoint
# at boot regardless of whether the visitor ever navigated to Live Demo.
# With lazy loading the Models / Analysis / About tabs cold-start in O(0)
# model-loads, and the Live Demo tab pays for exactly one model the first
# time it renders (both if the user enables compare mode).
#
# NOTE on `st.cache_resource`: the decorator keys caches by the *arguments*,
# so `_load_model_cached("MobileNetV2", "cpu")` and the "cuda" variant are
# separate cache entries. We pass `device.type` (a short string) rather than
# the `torch.device` object itself because torch.device isn't hashable in a
# way Streamlit's cache key hasher loves.


@st.cache_resource(show_spinner=False)
def _get_device() -> torch.device:
    """Pick the best device once per process, with CPU thread capping.

    Priority unchanged from `select_device`: CUDA > MPS > CPU. On CPU-only
    hosts (like Streamlit Community Cloud) we cap torch's intra-op thread
    pool so the shared container doesn't oversubscribe the CPU when
    multiple requests interleave. This bound was chosen empirically — too
    low starves single-image inference, too high fights neighbouring apps
    on the same host.
    """
    device = select_device(verbose=True)
    if device.type == "cpu":
        try:
            torch.set_num_threads(max(1, min(4, os.cpu_count() or 2)))
        except Exception:
            # set_num_threads raises if called after any inter-op has run;
            # silently tolerate because the default thread count is still
            # sensible on free-tier hosts.
            pass
    return device


@st.cache_resource(show_spinner=False)
def _available_model_names() -> list[str]:
    """Filesystem-only scan of registered model checkpoints. Safe at boot."""
    return list_available_models()


@st.cache_resource(show_spinner="Loading model…")
def _load_model_cached(name: str, device_type: str):
    """Instantiate + load one model on demand. Cached for the process lifetime.

    The Streamlit cache_resource decorator guarantees we only pay the model
    load cost once per (name, device) pair even if the user toggles between
    models or reruns the script by interacting with widgets.
    """
    return load_model_by_name(name, torch.device(device_type))


@st.cache_resource(show_spinner=False)
def _load_cifar10_test():
    return torchvision.datasets.CIFAR10(
        root="./data", train=False, download=True,
        transform=transforms.ToTensor(),
    )


@st.cache_resource(show_spinner=False)
def _load_confusion_matrices() -> dict | None:
    """Read the pre-computed confusion matrices JSON once per process.

    The matrices were computed from the verified checkpoints against the
    full 10 000-image CIFAR-10 test set (see results/confusion_matrices.json
    provenance note). Returning None lets callers hide the section gracefully
    if the file is missing — which happens on minimal clones / CI sandboxes
    and should never crash the Analysis tab.
    """
    path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "results",
        "confusion_matrices.json",
    )
    if not os.path.isfile(path):
        return None
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return None


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
        # Preserve active tab across theme changes (the theme bridge JS
        # may trigger a full page navigation that destroys session_state)
        current_nav = st.session_state.get("nav", NAV_OVERVIEW)
        st.query_params["tab"] = current_nav
        st.rerun()


def _render_runtime_panel(available_names: list[str], device: torch.device) -> None:
    """Shared runtime/status block used by both sidebar modes.

    `available_names` is the list of model keys whose checkpoints exist on
    disk, NOT the set of models currently loaded into memory. With lazy
    loading those are different: a model is only instantiated the first time
    the user actually runs inference on it. Surfacing the *available* count
    is the honest thing to show in a status panel.
    """
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

    st.sidebar.markdown(
        f"**Selectable models**  {len(available_names)} of {_TOTAL_TRAINED} architectures"
    )
    st.sidebar.caption(
        f"All {_TOTAL_TRAINED} architectures were trained and benchmarked; "
        f"{len(available_names)} have checkpoints present for live inference. "
        "See the Models tab for the full comparison."
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


def render_live_demo_sidebar(available_names: list[str], device: torch.device) -> dict:
    """Full control set. Only shown when the Live Demo tab is active."""
    st.sidebar.markdown("## Live demo controls")
    st.sidebar.caption("Configure the model and visualisation used on this tab.")

    if available_names:
        default = _recommended_model(available_names)
        # Annotate the options so the recommended model is visually obvious
        # in the dropdown itself. We keep the raw key as the return value and
        # only rewrite the label via format_func.
        def _label(name: str) -> str:
            if name == default:
                return f"{name} · recommended"
            return name

        # Guard against stale session_state (e.g. checkpoint removed between runs)
        if st.session_state.get("model_select") not in available_names:
            st.session_state.pop("model_select", None)

        selected_model = st.sidebar.selectbox(
            "Model",
            available_names,
            index=available_names.index(default),
            format_func=_label,
            key="model_select",
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
        if len(available_names) > 1:
            compare_mode = st.sidebar.checkbox(
                "Compare deployed models",
                value=False,
                help="Run every deployed model on the same image, side-by-side.",
            )
    else:
        selected_model = None
        top_k = 5
        show_gradcam = False
        gradcam_alpha = 0.5
        compare_mode = False

    st.sidebar.markdown("---")
    _render_runtime_panel(available_names, device)
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


def render_info_sidebar(available_names: list[str], device: torch.device) -> dict:
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
    _render_runtime_panel(available_names, device)
    st.sidebar.markdown("---")
    _render_theme_toggle()
    _render_sidebar_links()

    # Return default settings; these are only consumed by the Live Demo tab
    # anyway, so values here are effectively unused on other tabs.
    default_model = _recommended_model(available_names) if available_names else None
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


def render_overview_tab(available_names: list[str]) -> None:
    best_key = best_model_key()
    best = BENCHMARK_METRICS[best_key]
    deployed_count = len([k for k in available_names if k in BENCHMARK_METRICS])

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
        f'<div class="pred-conf">{top_conf:.1f}% confidence'
        f'{"  · closest CIFAR-10 class" if true_label is None else ""}'
        f'</div>'
        f'</div>',
        unsafe_allow_html=True,
    )

    # OOD caveat for user-uploaded images (no ground truth available)
    if true_label is None:
        st.warning(
            "**Closed-set classifier** — this model only knows "
            f"**10 CIFAR-10 classes** ({', '.join(CLASS_NAMES)}). "
            "If your image is not one of these, the prediction shows the "
            "**closest matching class**, not a correct identification. "
            "High confidence on out-of-distribution images is a known "
            "limitation of softmax classifiers.",
            icon="⚠️",
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
            # Transparent figure + theme-aware title color so the Grad-CAM
            # heatmap reads correctly in both light and dark modes. The
            # heatmap itself is `jet` regardless — that's the scientific
            # convention and doesn't depend on page background.
            # Use a neutral mid-gray that clears WCAG AA on both white
            # and dark backgrounds. Previously we picked different
            # colours per theme, but `theme == "auto"` fell through to
            # the dark branch and rendered near-invisible on real
            # light-mode pages. Slate-500 is the safe middle ground.
            _title_color = "#64748b"
            fig, ax = plt.subplots(figsize=(3, 3))
            fig.patch.set_alpha(0.0)
            ax.set_facecolor("none")
            ax.imshow(heatmap, cmap="jet")
            ax.axis("off")
            ax.set_title("Activation Map", fontsize=9, color=_title_color, pad=8)
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


def render_live_demo_tab(
    available_names: list[str],
    device: torch.device,
    settings: dict,
) -> None:
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

    # Show supported classes so users know what the model can recognise
    _class_chips = " ".join(
        f'<span class="chip">{c}</span>' for c in CLASS_NAMES
    )
    st.markdown(
        '<div style="margin-bottom:1rem;">'
        '<span style="font-size:0.72rem;font-weight:600;letter-spacing:0.1em;'
        'text-transform:uppercase;color:var(--text-muted);">Supported classes</span>'
        f'<br>{_class_chips}'
        '<p style="font-size:0.8rem;color:var(--text-dim);margin:0.4rem 0 0 0;">'
        'Images outside these 10 categories will be mapped to the closest class. '
        'See the Analysis tab for known confusion pairs.</p>'
        '</div>',
        unsafe_allow_html=True,
    )

    if not available_names:
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
        models_to_run = list(available_names)
    else:
        models_to_run = [settings["selected_model"]]

    # Lazy-load each selected model via the cached helper. First use of a
    # given model pays the load cost once; every subsequent rerun (slider
    # drag, preset click, Grad-CAM toggle) hits the in-process cache and
    # returns instantly. This is the key win over the old eager loader:
    # visitors who never touch Live Demo never pay for MobileNetV2 at all.
    def _get_model(name: str):
        try:
            return _load_model_cached(name, device.type)
        except FileNotFoundError as exc:
            st.error(f"Checkpoint missing for **{name}**: {exc}")
            return None
        except RuntimeError as exc:
            st.error(f"Checkpoint for **{name}** doesn't match the architecture: {exc}")
            return None

    if len(models_to_run) == 1:
        col_img, col_pred = st.columns([1, 2])
        with col_img:
            st.image(image, caption="Input image", width="stretch")
            if true_label is not None:
                st.caption(f"True label: **{true_label}**")
        with col_pred:
            name = models_to_run[0]
            model = _get_model(name)
            if model is not None:
                render_prediction(
                    model, name, image, device,
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
                model = _get_model(name)
                if model is not None:
                    render_prediction(
                        model, name, image, device,
                        settings["top_k"], settings["show_gradcam"],
                        settings["gradcam_alpha"], true_label,
                    )


# ============================================================================
#  Tab 3 — Model Comparison
# ============================================================================


def _render_pa_table(
    df: pd.DataFrame,
    *,
    progress_columns: dict[str, tuple[float, float, str]] | None = None,
    numeric_columns: dict[str, str] | None = None,
    center_columns: set[str] | None = None,
) -> None:
    """Render a pandas DataFrame as a themed HTML table.

    We replaced ``st.dataframe`` with this helper because Streamlit's
    DataFrame widget uses the Glide Data Grid, which rasterises every
    cell to a ``<canvas>``. Canvas pixels are outside the CSS cascade,
    so the cell backgrounds always rendered dark even when the rest of
    the page was in light mode. An HTML table is fully CSS-themeable
    and small reference tables don't need Glide's interactive features.

    Parameters
    ----------
    df:
        The DataFrame to render. Column order in ``df`` is preserved.
    progress_columns:
        Columns to render as inline progress bars, keyed by column name.
        Each value is a tuple of ``(min_value, max_value, format_string)``
        where ``format_string`` accepts a single ``{v}`` placeholder (e.g.
        ``"{v:.2f}%"`` or ``"{v:d}%"``).
    numeric_columns:
        Columns to right-align with a specific format string. The key is
        the column name and the value is a format string taking ``{v}``.
    center_columns:
        Columns to center-align.
    """
    progress_columns = progress_columns or {}
    numeric_columns = numeric_columns or {}
    center_columns = center_columns or set()

    import html as _html_mod

    head = "".join(f"<th>{_html_mod.escape(str(c))}</th>" for c in df.columns)
    body_rows: list[str] = []
    for _, row in df.iterrows():
        cells: list[str] = []
        for col in df.columns:
            val = row[col]
            if col in progress_columns:
                lo, hi, fmt = progress_columns[col]
                try:
                    num = float(val)
                except (TypeError, ValueError):
                    num = 0.0
                span = max(hi - lo, 1e-9)
                pct = max(0.0, min(100.0, 100.0 * (num - lo) / span))
                label = _html_mod.escape(fmt.format(v=num))
                cells.append(
                    '<td class="num">'
                    '<div class="pa-progress">'
                    f'<div class="fill" style="width:{pct:.2f}%;"></div>'
                    f'<span class="label">{label}</span>'
                    "</div></td>"
                )
            elif col in numeric_columns:
                try:
                    num = float(val)
                    text = numeric_columns[col].format(v=num)
                except (TypeError, ValueError):
                    text = str(val)
                cells.append(f'<td class="num">{_html_mod.escape(text)}</td>')
            elif col in center_columns:
                cells.append(f'<td class="center">{_html_mod.escape(str(val))}</td>')
            else:
                cells.append(f"<td>{_html_mod.escape(str(val))}</td>")
        body_rows.append("<tr>" + "".join(cells) + "</tr>")

    table_html = (
        '<table class="pa-table">'
        f"<thead><tr>{head}</tr></thead>"
        f"<tbody>{''.join(body_rows)}</tbody>"
        "</table>"
    )
    st.markdown(table_html, unsafe_allow_html=True)


def render_models_tab(available_names: list[str]) -> None:
    # Selectable count is derived from the filesystem check (available_names)
    # rather than the static `available` flag in BENCHMARK_METRICS, so it
    # accurately reflects what this deployment instance can actually load.
    selectable_count = len([k for k in available_names if k in BENCHMARK_METRICS])

    st.markdown('<div class="section-title">Model comparison</div>', unsafe_allow_html=True)
    st.markdown(
        f'<p class="section-sub">{_TOTAL_TRAINED} architectures were trained and '
        'evaluated head-to-head on the full 10 000-image CIFAR-10 test set. '
        f'{selectable_count} are selectable in the live demo — the rest are in the '
        'table for context but their checkpoints are not present in this deployment '
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
            "Deployed": "✓" if key in available_names else "—",
        })
    df = pd.DataFrame(rows)
    _render_pa_table(
        df,
        progress_columns={
            "Test Acc (%)": (0.0, 100.0, "{v:.2f}%"),
        },
        numeric_columns={
            "Latency (ms)": "{v:.2f} ms",
            "Size (MB)": "{v:.2f} MB",
            "Throughput (FPS)": "{v:.1f}",
        },
        center_columns={"Deployed", "Input"},
    )

    st.caption(
        "⭐ = highest-accuracy model on the test set. "
        "'Deployed' indicates whether the checkpoint ships with the live app. "
        "Non-deployed checkpoints were still trained and benchmarked; they are "
        "omitted from the demo for file-size and cold-start reasons on the "
        "free hosting tier, not because of missing work."
    )

    # ── Try a deployed model ────────────────────────────────────
    # Dynamic: any model in both available_names (filesystem check) AND
    # BENCHMARK_METRICS (has display info) gets a "Try" button. Models
    # without checkpoints on disk are listed as non-selectable context.
    _deployed_keys = [k for k in available_names if k in BENCHMARK_METRICS]
    _not_deployed_keys = [k for k in BENCHMARK_METRICS if k not in available_names]

    st.markdown(
        '<div class="section-title" style="margin-top:2rem;">Try a model</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<p class="section-sub">Deployed models can be tested on the Live Demo tab '
        'with your own images or CIFAR-10 samples.</p>',
        unsafe_allow_html=True,
    )

    _try_cols = st.columns(len(_deployed_keys))
    for _tc, _dk in zip(_try_cols, _deployed_keys):
        _dm = BENCHMARK_METRICS[_dk]
        with _tc:
            st.markdown(
                f'<div class="glass-card" style="text-align:center;">'
                f'<h3>{_dm["display_name"]}'
                f'{"  ⭐" if _dk == best_model_key() else ""}</h3>'
                f'<p style="margin:0.2rem 0;">'
                f'{_dm["test_accuracy"]:.2f}% accuracy · '
                f'{_dm["strategy"]}</p>'
                f'</div>',
                unsafe_allow_html=True,
            )

            def _go_try(model_key: str = _dk) -> None:
                st.session_state["nav"] = NAV_LIVE_DEMO
                st.session_state["model_select"] = model_key

            st.button(
                f"Try {_dm['display_name']} →",
                key=f"try_{_dk}",
                use_container_width=True,
                on_click=_go_try,
            )

    if _not_deployed_keys:
        st.markdown(
            '<p class="section-sub" style="margin-top:0.8rem;">'
            f'<b>{len(_not_deployed_keys)} additional model'
            f'{"s" if len(_not_deployed_keys) > 1 else ""}</b> '
            f'({", ".join(_not_deployed_keys)}) '
            'were trained and benchmarked but their checkpoints are not '
            'deployed — the free Streamlit Community Cloud tier has '
            'file-size and cold-start constraints that make shipping all '
            f'{_TOTAL_TRAINED} models impractical. Their metrics appear '
            'in the table above for context.</p>',
            unsafe_allow_html=True,
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

    # We render the convergence chart through matplotlib (not st.line_chart)
    # because Vega-Lite's generated SVG carries its own theme background
    # that CSS on the wrapper cannot touch — in light mode the chart area
    # stayed dark under a white page. Matplotlib lets us emit a fully
    # transparent figure that inherits the card surface, the same pattern
    # the confusion-matrix heatmap already uses.
    import matplotlib.pyplot as _cvg_plt

    _epochs = list(range(1, len(next(iter(CONVERGENCE_HISTORY.values()))) + 1))
    # Neutral slate tone for axes/labels — reads on both themes (WCAG AA).
    _cvg_axis = "#64748b"
    _cvg_grid = (100 / 255, 116 / 255, 139 / 255, 0.25)
    # Brand palette ordered for deterministic line assignment.
    _cvg_palette = ["#7c3aed", "#0ea5e9", "#10b981", "#f59e0b", "#ec4899"]

    _cvg_fig, _cvg_ax = _cvg_plt.subplots(figsize=(8.8, 3.6))
    _cvg_fig.patch.set_alpha(0.0)
    _cvg_ax.set_facecolor("none")

    for _i, (_name, _hist) in enumerate(CONVERGENCE_HISTORY.items()):
        _cvg_ax.plot(
            _epochs,
            _hist,
            label=_name,
            color=_cvg_palette[_i % len(_cvg_palette)],
            linewidth=2.2,
            marker="o",
            markersize=4,
        )

    _cvg_ax.set_xlabel("Epoch", color=_cvg_axis, fontsize=10)
    _cvg_ax.set_ylabel("Validation accuracy (%)", color=_cvg_axis, fontsize=10)
    _cvg_ax.set_ylim(0, 100)
    _cvg_ax.set_xticks(_epochs)
    _cvg_ax.tick_params(colors=_cvg_axis, labelsize=8)
    for _spine in _cvg_ax.spines.values():
        _spine.set_color(_cvg_grid)
    _cvg_ax.grid(True, color=_cvg_grid, linewidth=0.8)
    _legend = _cvg_ax.legend(
        loc="lower right",
        frameon=False,
        fontsize=9,
        labelcolor=_cvg_axis,
    )
    _cvg_fig.tight_layout()
    st.pyplot(_cvg_fig, clear_figure=True, use_container_width=True)
    _cvg_plt.close(_cvg_fig)


# ============================================================================
#  Tab 4 — Error Analysis
# ============================================================================

def _render_confusion_heatmap(
    matrix: list[list[int]],
    classes: list[str],
    title: str,
    theme_mode: str,
) -> None:
    """Render a 10×10 row-normalised confusion-matrix heatmap.

    Design choices:

    * Row normalisation (each row sums to 100 %) because CIFAR-10 is balanced
      but the reader is asking "given the true class, how often did the model
      predict each other class" — that's a per-row question, not a per-cell
      question. Raw counts hide the relative severity of misclassification
      on smaller classes; percentages do not.
    * Sequential colormap (matplotlib `Purples`) to match the brand palette
      and to stay readable on both light and dark themes (transparent fig
      background means cell luminance + auto text colour handles contrast).
    * Cell text is white on dark cells and near-black on light cells, driven
      by a luminance threshold. This is the standard readable-annotation
      pattern for heatmaps — eyeballing `cmap(0.5)` and picking manually is a
      guaranteed way to ship unreadable cells.
    """
    import matplotlib.pyplot as plt
    import matplotlib as mpl

    # Row-normalise to percentages. Zero-row guard in case a class is
    # somehow absent from the eval set (never happens on CIFAR-10 but the
    # divide-by-zero would be user-visible if it did).
    totals = [max(sum(row), 1) for row in matrix]
    norm = np.array([
        [100.0 * c / t for c in row]
        for row, t in zip(matrix, totals)
    ])

    # Neutral mid-gray that clears WCAG AA on both white and dark
    # backgrounds. Previously we branched on `theme_mode == "light"`,
    # but the default `"auto"` fell through to the else branch and
    # picked `#cbd5e1` (pale slate), which was effectively invisible
    # on the real light-mode confusion-matrix card. Slate-500 is the
    # safe middle ground for all three modes (light / dark / auto).
    _ = theme_mode  # retained in signature for API compat, intentionally unused
    axis_color = "#64748b"
    # Matplotlib does NOT parse CSS rgba() strings. We pass a 4-tuple of
    # floats in [0, 1] instead, which is the canonical matplotlib colour
    # format for an RGBA value. (100/255, 116/255, 139/255, 0.35).
    grid_color = (100 / 255, 116 / 255, 139 / 255, 0.35)

    fig, ax = plt.subplots(figsize=(5.2, 4.8))
    fig.patch.set_alpha(0.0)          # transparent — let the Streamlit card show
    ax.set_facecolor("none")

    # ``matplotlib.cm.get_cmap`` was deprecated in 3.7 and slated for
    # removal in 3.11. The replacement is the ``matplotlib.colormaps``
    # registry, which returns the same Colormap object.
    cmap = mpl.colormaps["Purples"]
    im = ax.imshow(norm, cmap=cmap, vmin=0, vmax=100, aspect="equal")

    ax.set_xticks(range(len(classes)))
    ax.set_yticks(range(len(classes)))
    ax.set_xticklabels(classes, rotation=45, ha="right", fontsize=8, color=axis_color)
    ax.set_yticklabels(classes, fontsize=8, color=axis_color)
    ax.set_xlabel("Predicted", fontsize=9, color=axis_color, labelpad=6)
    ax.set_ylabel("True",      fontsize=9, color=axis_color, labelpad=6)
    ax.set_title(title, fontsize=10, color=axis_color, pad=10)
    for spine in ax.spines.values():
        spine.set_color(grid_color)

    # Annotate each cell with its percentage. Auto-pick text colour from
    # the cell's luminance so the digits never vanish into the background.
    for i in range(len(classes)):
        for j in range(len(classes)):
            val = norm[i, j]
            rgba = cmap(val / 100.0)
            luminance = 0.2126 * rgba[0] + 0.7152 * rgba[1] + 0.0722 * rgba[2]
            text_color = "#ffffff" if luminance < 0.55 else "#1e1b4b"
            ax.text(
                j, i, f"{val:.0f}" if val >= 1 else "·",
                ha="center", va="center",
                fontsize=7, color=text_color,
            )

    cbar = fig.colorbar(im, ax=ax, shrink=0.85, pad=0.02)
    cbar.ax.tick_params(labelsize=7, colors=axis_color)
    cbar.outline.set_edgecolor(grid_color)
    cbar.set_label("% of true class", fontsize=8, color=axis_color)

    fig.tight_layout()
    st.pyplot(fig, width="stretch")
    plt.close(fig)


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
    _render_pa_table(
        df,
        progress_columns={
            "Reduction (%)": (0.0, 100.0, "{v:.0f}%"),
        },
        numeric_columns={
            "Custom CNN (errors)": "{v:.0f}",
            "MobileNetV2 (errors)": "{v:.0f}",
        },
    )

    # ── Full confusion matrix heatmap ─────────────────────────────
    # Row-normalised so each row reads as "given the true class, what does
    # the model predict?". The strong diagonal is the story: MobileNetV2's
    # transfer-learning head is almost entirely correct, with residual mass
    # concentrated in the cat↔dog quadrant and the vehicle pair. Data is
    # pre-computed from the verified checkpoint (see results/ JSON) so the
    # Analysis tab does not pay a full eval pass on cold-start.
    _cm_data = _load_confusion_matrices()
    if _cm_data is not None and "MobileNetV2" in _cm_data.get("matrices", {}):
        st.markdown(
            '<div class="section-title" style="margin-top:2rem;">Confusion matrix</div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            '<p class="section-sub">Row-normalised confusion matrix for MobileNetV2 on the '
            'full 10 000-image CIFAR-10 test set. Each row sums to 100 % of that class\'s '
            'true images; the diagonal is "got it right". The visible residual — a soft '
            'cat ↔ dog block and a dimmer truck ↔ automobile pair — matches the '
            'hardest-pair table above and confirms the model\'s errors are concentrated '
            'on genuinely similar classes, not random noise.</p>',
            unsafe_allow_html=True,
        )
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        _render_confusion_heatmap(
            matrix=_cm_data["matrices"]["MobileNetV2"],
            classes=list(_cm_data["classes"]),
            title="MobileNetV2 — row-normalised confusion matrix",
            theme_mode=st.session_state.get("theme", "auto"),
        )
        st.markdown('</div>', unsafe_allow_html=True)

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
    # IMPORTANT: keep this HTML flush-left (no leading indentation on
    # inner lines). CommonMark's indented-code-block rule turns any
    # block of 4+ leading spaces that follows a blank line into a
    # `<pre>` block, and since Streamlit's `st.markdown` runs the
    # content through markdown-it before the HTML sanitiser, indenting
    # the inner `<dt>`/`<dd>` lines caused everything after the first
    # `<dd>` to render as literal source in a dark syntax-highlighted
    # code block. No blank lines and no leading whitespace is the only
    # reliable fix; dedenting via textwrap also works but is easy to
    # accidentally reintroduce on a later edit.
    st.markdown(
        '<div class="model-card">'
        '<h3>CIFAR-10 classifier — portfolio study</h3>'
        '<dl>'
        '<dt>Training data</dt>'
        '<dd>CIFAR-10: 50 000 labelled 32×32 RGB images across 10 balanced '
        'classes (airplane, automobile, bird, cat, deer, dog, frog, horse, '
        'ship, truck). Reserved 10 000 images for testing.</dd>'
        '<dt>Intended use</dt>'
        '<dd>Educational exploration of transfer learning, interpretability, '
        'and ML-engineering discipline on a small-image benchmark. '
        'Demonstrates an end-to-end pipeline suitable for portfolio review.</dd>'
        '<dt>Out-of-scope uses</dt>'
        '<dd>Not intended for real-world image classification, content '
        'moderation, safety-critical systems, or any decision-making context '
        'where a wrong prediction has a meaningful cost. CIFAR-10 is a '
        'low-resolution research benchmark, not a production dataset.</dd>'
        '<dt>Known limitations</dt>'
        '<dd><ul style="margin:0.4rem 0 0 1.1rem; padding:0;">'
        '<li>32×32 training resolution — loses fine detail that real photos '
        'rely on.</li>'
        '<li>10 classes only — anything outside these classes gets '
        'force-mapped to the nearest lookalike (e.g. bicycle → automobile, '
        'wolf → dog).</li>'
        '<li>Frozen ImageNet features inherit whatever biases exist in '
        'ImageNet.</li>'
        '<li>Distribution shift: performance drops sharply on images that do '
        'not look like centred, cleanly-cropped, daylight CIFAR-10 samples.</li>'
        '</ul></dd>'
        '<dt>Evaluation</dt>'
        '<dd>All metrics on this site come from the full 10 000-image test '
        'set and are mirrored across README, training metadata JSON, and the '
        'comparison table — a single source of truth in '
        '<code>benchmark_data.py</code>.</dd>'
        '</dl>'
        '</div>',
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
            Hi, I'm <b>Pouya Alavi</b>. This is a small portfolio piece I keep
            coming back to whenever I want to sharpen one discipline of ML
            engineering in isolation — reproducible training runs, benchmarks
            I'm willing to defend, interpretability that a reviewer can
            actually inspect, and a delivery surface I'd be comfortable
            handing to a non-technical stakeholder. Everything on this page
            is computed from the same two checkpoints the Live Demo tab
            loads, so the numbers, the confusion matrix, and the Grad-CAM
            overlays all describe the <em>same</em> models — not a marketing
            version of them.
        </p>
        <p class="section-sub" style="margin-top:0.6rem;">
            I'm currently finishing a Bachelor of Information Technology at
            Macquarie University (AI and Web/App Development majors) and
            looking for early-career roles where the same disciplines
            matter.
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
        <div class="glass-card contact-card">
            <p>
                📧 <a href="mailto:pouya@pouyaalavi.dev">pouya@pouyaalavi.dev</a><br>
                🌐 <a href="https://www.pouyaalavi.dev" target="_blank">pouyaalavi.dev</a><br>
                💼 <a href="https://www.linkedin.com/in/pouya-alavi" target="_blank">linkedin.com/in/pouya-alavi</a><br>
                🐙 <a href="https://github.com/mrpouyaalavi" target="_blank">github.com/mrpouyaalavi</a><br>
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
    # Lazy boot: we resolve the device and *list* the available checkpoints,
    # but we do NOT load any model into memory yet. The Live Demo tab is the
    # only place model weights are actually needed, and it loads on demand
    # via `_load_model_cached()`. On tabs that don't use the model (Overview,
    # Models, Analysis, About) this gets us O(0) model-loads at cold start.
    device = _get_device()
    available_names = _available_model_names()

    # Restore navigation from URL query params ONLY on fresh page loads (when
    # _nav_restored is absent because session_state was destroyed by a full
    # browser navigation — e.g. the theme bridge JS adding ?theme= to the
    # URL).  On normal Streamlit reruns (widget clicks, st.rerun()), the
    # segmented_control widget is the authoritative source: its on-change
    # callback writes directly to st.session_state["nav"].  Unconditionally
    # reading st.query_params here would OVERWRITE that value with the stale
    # URL (which hasn't been synced yet) and lock the tabs on Overview.
    if "_nav_restored" not in st.session_state:
        st.session_state["_nav_restored"] = True
        url_tab = st.query_params.get("tab")
        if url_tab and url_tab in NAV_OPTIONS:
            st.session_state["nav"] = url_tab

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
        settings = render_live_demo_sidebar(available_names, device)
    else:
        settings = render_info_sidebar(available_names, device)

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

    # Keep the URL query param in sync so the tab survives full page reloads
    if active != st.query_params.get("tab"):
        st.query_params["tab"] = active

    if active == NAV_OVERVIEW:
        render_overview_tab(available_names)
    elif active == NAV_LIVE_DEMO:
        render_live_demo_tab(available_names, device, settings)
    elif active == NAV_MODELS:
        render_models_tab(available_names)
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
