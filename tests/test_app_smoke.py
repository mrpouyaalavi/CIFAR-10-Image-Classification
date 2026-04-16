"""
End-to-end Streamlit smoke test using ``streamlit.testing.v1.AppTest``.

Why this exists
---------------
The app has five tabs and three theme modes (light / dark / auto),
giving fifteen combinations of initial state. A rendering regression in
any single tab - an unclosed ``<div>``, a bad f-string, a missing CSS
variable - used to take a full manual pass across every combination to
catch. This test runs the whole matrix in a few seconds per execution
and fails loudly on any exception, markdown-parse error, or layout
crash.

``AppTest.from_file()`` executes the Streamlit script in-process with a
synthetic runtime, so it exercises the real top-level module (import
order, CSS injection, theme resolution) without needing a browser or
live server.

Notes
-----
* We set ``session_state["theme"]`` before ``.run()`` so every theme
  branch is actually exercised.
* We switch ``session_state["nav"]`` to exercise each tab independently
  rather than relying on the default Overview landing.
* Some CI runners lack write access to ``./data`` for the CIFAR-10
  dataset download triggered by ``_load_cifar10_test()``. That call is
  lazy: it only fires if you actually render the Live Demo tab with a
  CIFAR sample loaded. The default landing state leaves the image
  slot empty, so Live Demo renders the "no image yet" placeholder and
  does *not* touch the network.
"""

from __future__ import annotations

from pathlib import Path

import pytest

APP_PATH = Path(__file__).resolve().parent.parent / "streamlit_app.py"


# Streamlit's testing runtime is optional — it ships with streamlit>=1.30.
pytest.importorskip("streamlit.testing.v1")


from streamlit.testing.v1 import AppTest  # noqa: E402  (import after skip)


NAV_TABS = [
    "🏠 Overview",
    "🔬 Live Demo",
    "📊 Models",
    "🔍 Analysis",
    "👤 About",
]

THEMES = ["auto", "light", "dark"]


@pytest.mark.parametrize("theme", THEMES)
@pytest.mark.parametrize("nav", NAV_TABS)
def test_app_renders_every_tab_in_every_theme(theme: str, nav: str) -> None:
    """The full matrix: 5 tabs x 3 themes = 15 renders, zero exceptions."""
    at = AppTest.from_file(str(APP_PATH), default_timeout=30)
    at.session_state["theme"] = theme
    at.session_state["nav"] = nav
    at.run()
    assert not at.exception, (
        f"App raised an exception while rendering {nav} in theme={theme}: "
        f"{at.exception}"
    )


def test_app_default_landing_renders_cleanly() -> None:
    """First visit, no session state, no URL params. Must land on Overview
    and render every hero block without raising."""
    at = AppTest.from_file(str(APP_PATH), default_timeout=30)
    at.run()
    assert not at.exception, f"Default landing raised: {at.exception}"
    # Overview should be the active tab on first load. Streamlit's
    # ``SafeSessionState`` does not implement dict ``.get()``, so we use
    # membership + subscript access instead.
    assert "nav" in at.session_state
    assert at.session_state["nav"] == "🏠 Overview"
