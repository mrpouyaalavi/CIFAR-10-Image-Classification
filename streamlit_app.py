"""
CIFAR-10 Image Classification — Streamlit Landing Page
=======================================================

This app has been migrated to Hugging Face Spaces (Gradio).
This Streamlit page serves as a professional landing / gateway
so that the original public URL does not break:

    https://cifar10-pouyaalavi.streamlit.app/

Visitors are directed to the new Hugging Face Space for the
full interactive demo.

Author : Pouya Alavi  (pouya@pouyaalavi.dev)
License: MIT
"""

from __future__ import annotations

import os
import streamlit as st

# ── Page Config ─────────────────────────────────────────────────────────────

_FAVICON_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "assets", "favicon.svg"
)
_page_icon = _FAVICON_PATH if os.path.isfile(_FAVICON_PATH) else "🧠"

st.set_page_config(
    page_title="CIFAR-10 Classifier — Pouya Alavi",
    page_icon=_page_icon,
    layout="centered",
    initial_sidebar_state="collapsed",
)

# ── Custom CSS ──────────────────────────────────────────────────────────────

st.markdown(
    """
    <style>
    /* Hide Streamlit UI chrome for a cleaner landing page */
    #MainMenu, header, footer { visibility: hidden; }

    .landing-container {
        max-width: 680px;
        margin: 2rem auto;
        text-align: center;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    }
    .landing-container h1 {
        font-size: 2.4rem;
        margin-bottom: 0.25rem;
        color: #1a1a2e;
    }
    .landing-container .tagline {
        font-size: 1.1rem;
        color: #555;
        margin-bottom: 2rem;
    }
    .landing-container .card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 16px;
        padding: 2.5rem 2rem;
        color: white;
        margin-bottom: 2rem;
        box-shadow: 0 8px 30px rgba(102, 126, 234, 0.25);
    }
    .landing-container .card h2 {
        margin-top: 0;
        font-size: 1.5rem;
    }
    .landing-container .card p {
        font-size: 1rem;
        line-height: 1.6;
        opacity: 0.95;
    }
    .cta-button {
        display: inline-block;
        background: white;
        color: #764ba2;
        font-weight: 700;
        font-size: 1.15rem;
        padding: 0.85rem 2.4rem;
        border-radius: 50px;
        text-decoration: none;
        margin-top: 1rem;
        transition: transform 0.2s, box-shadow 0.2s;
        box-shadow: 0 4px 14px rgba(0,0,0,0.15);
    }
    .cta-button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(0,0,0,0.2);
        color: #764ba2;
    }
    .metrics-row {
        display: flex;
        justify-content: center;
        gap: 2rem;
        margin: 1.5rem 0 2rem;
        flex-wrap: wrap;
    }
    .metric-box {
        background: #f8f9fa;
        border-radius: 12px;
        padding: 1.2rem 1.5rem;
        min-width: 140px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.06);
    }
    .metric-box .value {
        font-size: 1.8rem;
        font-weight: 800;
        color: #764ba2;
    }
    .metric-box .label {
        font-size: 0.85rem;
        color: #888;
        margin-top: 0.2rem;
    }
    .footer-links {
        margin-top: 2rem;
        font-size: 0.9rem;
        color: #888;
    }
    .footer-links a {
        color: #667eea;
        text-decoration: none;
    }
    .footer-links a:hover { text-decoration: underline; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ── Landing Page Content ────────────────────────────────────────────────────

# TODO: Replace this placeholder with the actual Hugging Face Space URL
# once the Space is created and deployed.
HF_SPACE_URL = "https://huggingface.co/spaces/mrpouyaalavi/CIFAR-10-Image-Classification"

st.markdown(
    f"""
    <div class="landing-container">

        <h1>🧠 CIFAR-10 Image Classification</h1>
        <p class="tagline">
            Custom CNN vs Transfer Learning — A Comparative Deep Learning Study
        </p>

        <div class="metrics-row">
            <div class="metric-box">
                <div class="value">86.91%</div>
                <div class="label">Best Accuracy (MobileNetV2)</div>
            </div>
            <div class="metric-box">
                <div class="value">5</div>
                <div class="label">Architectures Studied</div>
            </div>
            <div class="metric-box">
                <div class="value">192&times;</div>
                <div class="label">Fewer Trainable Params</div>
            </div>
        </div>

        <div class="card">
            <h2>🚀 Demo Has Moved</h2>
            <p>
                The interactive demo is now hosted on
                <strong>Hugging Face Spaces</strong> with a faster Gradio interface,
                side-by-side model comparison, and example images.
            </p>
            <a class="cta-button" href="{HF_SPACE_URL}" target="_blank">
                Open Live Demo &rarr;
            </a>
        </div>

        <div class="footer-links">
            <a href="https://github.com/mrpouyaalavi/CIFAR-10-Image-Classification">
                GitHub Repository
            </a>
            &nbsp;&middot;&nbsp;
            <a href="https://pouyaalavi.dev">Portfolio</a>
            &nbsp;&middot;&nbsp;
            Built with PyTorch &middot; Gradio &middot; Hugging Face Spaces
        </div>

    </div>
    """,
    unsafe_allow_html=True,
)
