"""
CIFAR-10 Image Classification — Streamlit Landing Page
======================================================

This Streamlit app acts as a lightweight landing page for the original public URL:

    https://cifar10-pouyaalavi.streamlit.app/

The main interactive demo has been migrated to Hugging Face Spaces using Gradio.
This page preserves the old Streamlit link and redirects visitors to the new live demo.

Author : Pouya Alavi  (pouya@pouyaalavi.dev)
License: MIT
"""

from __future__ import annotations

import os
import streamlit as st

# -----------------------------------------------------------------------------
# Page Configuration
# -----------------------------------------------------------------------------

HF_SPACE_URL = "https://mrpouyaalavi-cifar-10-image-classification.hf.space"
GITHUB_URL = "https://github.com/mrpouyaalavi/CIFAR-10-Image-Classification"
PORTFOLIO_URL = "https://pouyaalavi.dev"

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

# -----------------------------------------------------------------------------
# Custom CSS
# -----------------------------------------------------------------------------

st.markdown(
    """
    <style>
    #MainMenu, header, footer {
        visibility: hidden;
    }

    .block-container {
        padding-top: 1.5rem;
        padding-bottom: 2rem;
    }

    .landing-container {
        max-width: 760px;
        margin: 0 auto;
        text-align: center;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    }

    .landing-container h1 {
        font-size: 2.5rem;
        margin-bottom: 0.35rem;
        color: #1f2937;
        line-height: 1.2;
    }

    .landing-container .tagline {
        font-size: 1.08rem;
        color: #6b7280;
        margin-bottom: 2rem;
    }

    .card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 18px;
        padding: 2.4rem 2rem;
        color: #ffffff;
        margin: 1.5rem 0 2rem;
        box-shadow: 0 10px 30px rgba(102, 126, 234, 0.22);
    }

    .card h2 {
        margin-top: 0;
        margin-bottom: 0.75rem;
        font-size: 1.6rem;
        color: #ffffff;
    }

    .card p {
        font-size: 1rem;
        line-height: 1.7;
        margin: 0 auto;
        max-width: 560px;
        opacity: 0.98;
    }

    .cta-button {
        display: inline-block;
        background: #ffffff;
        color: #6d3fc0 !important;
        font-weight: 700;
        font-size: 1.05rem;
        padding: 0.9rem 2.2rem;
        border-radius: 999px;
        text-decoration: none;
        margin-top: 1.35rem;
        transition: transform 0.18s ease, box-shadow 0.18s ease;
        box-shadow: 0 6px 18px rgba(0, 0, 0, 0.18);
    }

    .cta-button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 22px rgba(0, 0, 0, 0.22);
        color: #6d3fc0 !important;
        text-decoration: none;
    }

    .metrics-row {
        display: flex;
        justify-content: center;
        gap: 1rem;
        margin: 1.4rem 0 2rem;
        flex-wrap: wrap;
    }

    .metric-box {
        background: #f8fafc;
        border: 1px solid #e5e7eb;
        border-radius: 14px;
        padding: 1.15rem 1.4rem;
        min-width: 155px;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.05);
    }

    .metric-box .value {
        font-size: 1.7rem;
        font-weight: 800;
        color: #6d3fc0;
    }

    .metric-box .label {
        font-size: 0.88rem;
        color: #6b7280;
        margin-top: 0.2rem;
        line-height: 1.35;
    }

    .note {
        margin-top: 1rem;
        color: #6b7280;
        font-size: 0.98rem;
    }

    .footer-links {
        margin-top: 2rem;
        font-size: 0.93rem;
        color: #6b7280;
        line-height: 1.7;
    }

    .footer-links a {
        color: #667eea;
        text-decoration: none;
        font-weight: 600;
    }

    .footer-links a:hover {
        text-decoration: underline;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# -----------------------------------------------------------------------------
# Auto Redirect
# -----------------------------------------------------------------------------

st.markdown(
    f"""
    <meta http-equiv="refresh" content="2; url={HF_SPACE_URL}">
    <script>
        window.setTimeout(function() {{
            window.location.href = "{HF_SPACE_URL}";
        }}, 2000);
    </script>
    """,
    unsafe_allow_html=True,
)

# -----------------------------------------------------------------------------
# Landing Page Content
# -----------------------------------------------------------------------------

st.markdown(
    f"""
    <div class="landing-container">

        <h1>🧠 CIFAR-10 Image Classification</h1>
        <p class="tagline">
            Custom CNN vs MobileNetV2 vs ResNet-18 — A polished deep learning comparison project
        </p>

        <div class="metrics-row">
            <div class="metric-box">
                <div class="value">86.91%</div>
                <div class="label">Best Accuracy<br>(MobileNetV2)</div>
            </div>
            <div class="metric-box">
                <div class="value">3</div>
                <div class="label">Models<br>Compared</div>
            </div>
            <div class="metric-box">
                <div class="value">PyTorch</div>
                <div class="label">Inference &<br>Training Stack</div>
            </div>
        </div>

        <div class="card">
            <h2>🚀 Demo Has Moved</h2>
            <p>
                The full interactive demo is now hosted on <strong>Hugging Face Spaces</strong>
                with a cleaner Gradio interface, faster experience, and support for comparing
                multiple model architectures in one place.
            </p>
            <a class="cta-button" href="{HF_SPACE_URL}" target="_self">
                Open Live Demo →
            </a>
        </div>

        <p class="note">
            You will be redirected automatically in a moment.
            If that does not happen, use the button above.
        </p>

        <div class="footer-links">
            <a href="{GITHUB_URL}" target="_blank">GitHub Repository</a>
            &nbsp;&middot;&nbsp;
            <a href="{PORTFOLIO_URL}" target="_blank">Portfolio</a>
            <br />
            Built with PyTorch, Gradio, and Hugging Face Spaces
        </div>

    </div>
    """,
    unsafe_allow_html=True,
)