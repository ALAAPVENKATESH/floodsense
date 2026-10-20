"""Shared page header and the missing-model state."""

from __future__ import annotations

from app.ui.components import empty_state, page_header, render, status_chips
from app.ui.state import model, metrics, source_name, tree_count


def header(title: str, description: str) -> dict | None:
    """Draw the page title and status chips. Returns the metrics dict when present."""
    report = metrics()
    render(page_header(title, description, status_chips(source_name(report), tree_count(report))))
    return report


def require_model():
    """Return the fitted model, or show the next command and return None."""
    fitted = model()
    if fitted is None:
        render(
            empty_state(
                "Model not trained",
                "From the project folder run python scripts/build_dataset.py, then python scripts/train.py.",
            )
        )
        return None
    return fitted
