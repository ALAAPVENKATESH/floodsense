"""FloodSense entrypoint. The model implementation lives in src/flood_rf."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
for path in (str(ROOT), str(SRC)):
    if path not in sys.path:
        sys.path.insert(0, path)

import streamlit as st

import config
from app.ui.components import banner, footer, logo_lockup, render
from app.ui.nav import bind, go
from app.ui.state import metrics, source_name, trained_at, tree_count
from app.ui.styles import inject_styles
from app.ui.theme import NAV
from app.views import about, batch, dashboard, dataset, insights, predict

_ICON = ROOT / "app" / "assets" / "mark.png"


def main() -> None:
    """Draw the shell, then the current page."""
    st.set_page_config(
        page_title="FloodSense",
        page_icon=str(_ICON) if _ICON.exists() else "💧",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    st.session_state.setdefault("history", [])
    inject_styles(_theme_mode())
    pages = {
        "dashboard": st.Page(dashboard.page, title="Dashboard", url_path="dashboard", default=True),
        "predict": st.Page(predict.page, title="Predict", url_path="predict"),
        "batch": st.Page(batch.page, title="Batch", url_path="batch"),
        "insights": st.Page(insights.page, title="Insights", url_path="insights"),
        "dataset": st.Page(dataset.page, title="Dataset", url_path="dataset"),
        "about": st.Page(about.page, title="About", url_path="about"),
    }
    bind(pages)
    current = st.navigation(list(pages.values()), position="hidden")
    wanted = st.query_params.get("page")
    if isinstance(wanted, list):
        wanted = wanted[0] if wanted else None
    if wanted in pages and current.url_path != wanted:
        go(wanted)
    _sidebar(current)
    _banner()
    render(footer())
    current.run()


def _theme_mode() -> str:
    """Light or dark. A page change can reset the radio; the saved choice wins."""
    saved = st.session_state.get("theme_choice")
    if saved in {"Light", "Dark"} and st.session_state.get("theme_name") != saved:
        st.session_state["theme_name"] = saved
    selected = st.session_state.get("theme_name", saved or "Light")
    return "dark" if selected == "Dark" else "light"


def _remember_theme() -> None:
    st.session_state["theme_choice"] = st.session_state.get("theme_name", "Light")


def _sidebar(current) -> None:
    report = metrics()
    source = source_name(report)
    trees = tree_count(report)
    with st.sidebar:
        render(logo_lockup())
        render("<div class='fs-nav-label'>Navigate</div>")
        path = current.url_path or "dashboard"
        for key, label, icon in NAV:
            active = path == key
            clicked = st.button(
                f"{icon}  {label}",
                key=f"nav_{key}",
                use_container_width=True,
                type="primary" if active else "secondary",
            )
            if clicked and not active:
                go(key)
        render("<hr class='fs-hr'/>")
        kind = "good" if source == "REAL" else "warn"
        render(f"<span class='fs-chip fs-chip-{kind}'>{source}</span>")
        count = f"{trees} trees" if trees else "model not trained"
        render(f"<div class='fs-side-meta'>Random Forest · {count}<br>Trained {trained_at()}</div>")
        render("<div class='fs-nav-label'>Appearance</div>")
        st.radio(
            "Theme",
            ["Light", "Dark"],
            horizontal=True,
            key="theme_name",
            label_visibility="collapsed",
            on_change=_remember_theme,
        )


def _banner() -> None:
    report = metrics()
    source = source_name(report)
    if report is None or source == "REAL":
        return
    render(
        banner(
            "Simulated for coursework. Not measured rainfall, terrain, or flood history. "
            "Do not report these metrics as real-world performance."
        )
    )


if __name__ == "__main__":
    main()
