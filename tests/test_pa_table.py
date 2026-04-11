"""
Tests for the themed HTML table helper in ``app.py``.

We introduced ``_render_pa_table()`` to replace Streamlit's canvas-based
DataFrame widget, which couldn't be themed in light mode. This test
suite pins the helper's HTML output so a refactor can't silently break
* the progress-bar rendering used for "Test Acc" / "Reduction"
* the numeric formatting for latency / size / throughput
* HTML escaping (defense against injection through dataset columns)
"""

from __future__ import annotations

import pandas as pd
import pytest
import streamlit as st

import app  # noqa: F401 — import for side-effects (registers helper)


@pytest.fixture(autouse=True)
def capture_markdown(monkeypatch):
    """Monkeypatch ``st.markdown`` to capture the emitted HTML.

    We don't actually care about rendering to a page — only that the
    helper produces the right HTML. Using a fixture keeps each test
    isolated and avoids Streamlit's global runtime bookkeeping.
    """
    captured: list[str] = []

    def _fake_markdown(body: str, *args, **kwargs) -> None:
        captured.append(body)

    monkeypatch.setattr(st, "markdown", _fake_markdown)
    return captured


def test_pa_table_progress_column_renders_fill_bar(capture_markdown) -> None:
    df = pd.DataFrame(
        {"Model": ["Alpha", "Beta"], "Score": [25.0, 75.0]}
    )
    app._render_pa_table(
        df,
        progress_columns={"Score": (0.0, 100.0, "{v:.1f}%")},
    )
    assert capture_markdown, "helper did not emit markdown"
    html = capture_markdown[0]
    assert 'class="pa-table"' in html
    assert "pa-progress" in html
    # Fill width should match the score percentage for each row
    assert "width:25.00%" in html
    assert "width:75.00%" in html
    # Formatted label should appear, escaped by ``html.escape``
    assert "25.0%" in html
    assert "75.0%" in html


def test_pa_table_numeric_column_right_aligned(capture_markdown) -> None:
    df = pd.DataFrame({"Model": ["Alpha"], "Latency": [17.22]})
    app._render_pa_table(
        df,
        numeric_columns={"Latency": "{v:.2f} ms"},
    )
    html = capture_markdown[0]
    assert 'class="num"' in html
    assert "17.22 ms" in html


def test_pa_table_center_column_has_center_class(capture_markdown) -> None:
    df = pd.DataFrame({"Model": ["Alpha"], "Deployed": ["✓"]})
    app._render_pa_table(df, center_columns={"Deployed"})
    html = capture_markdown[0]
    assert 'class="center"' in html
    assert "✓" in html


def test_pa_table_escapes_html_injection(capture_markdown) -> None:
    """A dataset column containing ``<script>`` must not be rendered as
    a raw script tag — ``_render_pa_table`` runs every value through
    ``html.escape`` for defense in depth."""
    df = pd.DataFrame({"Model": ["<script>alert(1)</script>"]})
    app._render_pa_table(df)
    html = capture_markdown[0]
    assert "<script>alert(1)</script>" not in html
    assert "&lt;script&gt;" in html


def test_pa_table_preserves_column_order(capture_markdown) -> None:
    df = pd.DataFrame(
        {"Z": [1], "A": [2], "M": [3]},
    )
    app._render_pa_table(df)
    html = capture_markdown[0]
    z_pos = html.find("<th>Z</th>")
    a_pos = html.find("<th>A</th>")
    m_pos = html.find("<th>M</th>")
    assert 0 <= z_pos < a_pos < m_pos
