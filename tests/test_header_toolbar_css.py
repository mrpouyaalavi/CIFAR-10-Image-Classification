"""Regression tests for the top-right Streamlit header toolbar styling."""

from __future__ import annotations

from pathlib import Path

import pytest

APP_PATH = Path(__file__).resolve().parent.parent / "streamlit_app.py"

pytest.importorskip("streamlit.testing.v1")

from streamlit.testing.v1 import AppTest  # noqa: E402


def test_light_theme_css_keeps_header_icons_visible_and_clickable() -> None:
    """The light-theme stylesheet must repaint and surface header icons."""
    at = AppTest.from_file(str(APP_PATH), default_timeout=30)
    at.session_state["theme"] = "light"
    at.run()

    assert not at.exception, f"App raised while building CSS: {at.exception}"
    css = at.markdown[0].value

    assert '[data-testid="stToolbarActionButton"] svg path' in css
    assert "stroke: currentColor !important;" in css
    assert "visibility: visible !important;" in css
    assert "opacity: 1 !important;" in css
    assert "z-index: 1000 !important;" in css
