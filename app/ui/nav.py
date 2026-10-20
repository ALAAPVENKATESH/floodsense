"""Page registry so a view can switch pages without importing the entrypoint."""

from __future__ import annotations

import streamlit as st

_pages: dict = {}


def bind(pages: dict) -> None:
    """Store the st.Page objects created in the entrypoint."""
    _pages.clear()
    _pages.update(pages)


def go(key: str) -> None:
    """Open a registered page."""
    st.switch_page(_pages[key])
