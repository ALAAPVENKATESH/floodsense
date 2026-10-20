"""HTML components for FloodSense. Widgets stay in the page modules."""

from __future__ import annotations

from html import escape

import streamlit as st

_LOGO = """
<svg width="32" height="32" viewBox="0 0 32 32" aria-hidden="true">
  <rect width="32" height="32" rx="8" fill="#4F46E5"/>
  <path d="M16 6c3.2 4.2 6 7.2 6 11.2a6 6 0 1 1-12 0C10 13.2 12.8 10.2 16 6z" fill="#fff"/>
  <path d="M16 16.2c1.4 1.6 2.2 2.6 2.2 3.8a2.2 2.2 0 1 1-4.4 0c0-1.2.8-2.2 2.2-3.8z" fill="#4F46E5"/>
</svg>
"""


def render(html: str) -> None:
    """Write a trusted HTML fragment. Callers escape any user text first."""
    st.markdown(html, unsafe_allow_html=True)


def logo_lockup() -> str:
    """Sidebar mark plus the product name."""
    return (
        "<div class='fs-brand'>"
        f"{_LOGO}"
        "<div><div class='fs-brand-name'>FloodSense</div>"
        "<div class='fs-brand-tag'>Urban flood-risk intelligence</div></div></div>"
    )


def page_header(title: str, description: str, chips: list[tuple[str, str]]) -> str:
    """Page title, one-line description, and status chips."""
    chip_html = "".join(
        f"<span class='fs-chip fs-chip-{escape(kind)}'>{escape(text)}</span>"
        for text, kind in chips
    )
    return (
        "<header class='fs-page-head'><div>"
        f"<h1 class='fs-page-title'>{escape(title)}</h1>"
        f"<p class='fs-page-desc'>{escape(description)}</p></div>"
        f"<div class='fs-chips'>{chip_html}</div></header>"
    )


def status_chips(source: str, n_trees: int | None) -> list[tuple[str, str]]:
    """Right-aligned chips shared by every page."""
    trees = f"{n_trees} trees" if n_trees else "unavailable"
    kind = "good" if source == "REAL" else "warn"
    label = source if source else "UNKNOWN"
    return [
        (f"Model: Random Forest · {trees}", "neutral"),
        (f"Data: {label}", kind),
    ]


def banner(text: str) -> str:
    """Persistent amber notice. The mark is a label, not a close control."""
    return (
        "<div class='fs-banner'><div>"
        "<strong>Synthetic dataset</strong>"
        f"<span>{escape(text)}</span></div>"
        "<div class='fs-banner-x' title='This notice stays on every page'>×</div></div>"
    )


def kpi_card(label: str, value: str, delta: str) -> str:
    """One metric tile."""
    return (
        "<div class='fs-kpi'>"
        f"<div class='fs-kpi-label'>{escape(label)}</div>"
        f"<div class='fs-kpi-value'>{escape(value)}</div>"
        f"<div class='fs-kpi-delta'>{escape(delta)}</div></div>"
    )


def section_header(title: str, note: str = "") -> str:
    """Heading inside a card."""
    note_html = f"<p class='fs-section-note'>{escape(note)}</p>" if note else ""
    return f"<h2 class='fs-section'>{escape(title)}</h2>{note_html}"


def badge(label: str) -> str:
    """Risk pill. Unknown labels get a neutral chip."""
    safe = escape(label)
    if label in {"LOW", "MODERATE", "HIGH"}:
        return f"<span class='fs-badge fs-badge-{safe}'>{safe}</span>"
    return f"<span class='fs-chip'>{safe}</span>"


def risk_result_card(label: str, confidence: float, vote_text: str) -> str:
    """Large predicted class, badge, and vote sentence."""
    from app.ui.theme import RISK_COLORS

    color = RISK_COLORS.get(label, "#4F46E5")
    return (
        "<div class='fs-risk'>"
        "<div class='fs-risk-kicker'>Predicted risk</div>"
        "<div class='fs-risk-row'>"
        f"<div class='fs-risk-class' style='color:{color}'>{escape(label)}</div>"
        f"{badge(label)}</div>"
        f"<div class='fs-risk-sub'>{escape(f'{confidence:.0f}% confidence · {vote_text}')}</div>"
        "</div>"
    )


def empty_state(title: str, body: str) -> str:
    """Friendly placeholder with the next action in the body text."""
    return (
        "<div class='fs-empty'>"
        f"<div class='fs-empty-title'>{escape(title)}</div>"
        f"<p>{escape(body)}</p></div>"
    )


def alert(text: str) -> str:
    """Inline validation message."""
    return f"<div class='fs-alert'>{escape(text)}</div>"


def feature_pill(text: str) -> str:
    """Small stack or topic chip."""
    return f"<span class='fs-pill'>{escape(text)}</span>"


def callout(text: str) -> str:
    """Short caveat or causation note."""
    return f"<div class='fs-callout'>{escape(text)}</div>"


def footer() -> str:
    """Fixed academic disclaimer."""
    return (
        "<div class='fs-footer'>Academic prediction system. "
        "Not an official emergency-warning or evacuation tool.</div>"
    )


def data_table(columns: list[str], rows: list[list[str]], badges: dict[int, str] | None = None) -> str:
    """Compact HTML table. ``badges`` maps a column index to a risk label already stored in the cell."""
    badges = badges or {}
    head = "".join(f"<th>{escape(name)}</th>" for name in columns)
    body = []
    for row in rows:
        cells = []
        for index, value in enumerate(row):
            if index in badges:
                cells.append(f"<td>{badge(value)}</td>")
            else:
                cells.append(f"<td>{escape(str(value))}</td>")
        body.append("<tr>" + "".join(cells) + "</tr>")
    return f"<table class='fs-table'><thead><tr>{head}</tr></thead><tbody>{''.join(body)}</tbody></table>"


def tree_block(lines: list[tuple[int, str]]) -> str:
    """Indented read-only view of the top of one tree."""
    rows = []
    for depth, text in lines:
        rows.append(f"<div class='fs-node' style='margin-left:{depth * 16}px'>{escape(text)}</div>")
    return "".join(rows) or "<p class='fs-section-note'>This tree has no nodes.</p>"


def pipeline() -> str:
    """The project pipeline as a wrapping row of steps."""
    from app.ui.theme import PIPELINE

    items = "".join(f"<li>{escape(step)}</li>" for step in PIPELINE)
    return f"<ol class='fs-pipeline'>{items}</ol>"
