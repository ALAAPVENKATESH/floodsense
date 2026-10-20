"""FloodSense home: holdout KPIs, charts, and recent scores."""

from __future__ import annotations

import streamlit as st

from app.ui import charts
from app.ui.components import data_table, empty_state, kpi_card, render
from app.ui.nav import go
from app.ui.state import mode
from app.ui.theme import chart_theme
from app.views._frame import header, require_model


def page() -> None:
    """Dashboard."""
    report = header(
        "Dashboard",
        "Holdout quality of the self-implemented forest, and scores from this session.",
    )
    theme = chart_theme(mode())
    with st.container(border=True):
        left, right = st.columns([4, 1], gap="medium", vertical_alignment="center")
        with left:
            render(
                "<div class='fs-kicker'>FloodSense</div>"
                "<h2 class='fs-section' style='font-size:28px;letter-spacing:-0.03em;margin:6px 0 8px'>Urban flood-risk intelligence</h2>"
                "<p class='fs-page-desc'>Score a location as low, moderate, or high risk from rainfall, terrain, "
                "drainage, and flood history. The forest in this app is implemented in the project.</p>"
            )
        with right:
            if st.button("Run a prediction", type="primary", use_container_width=True):
                go("predict")
    if require_model() is None:
        return
    if report is None:
        render(empty_state("Metrics missing", "Run python scripts/train.py to write models/metrics.json."))
        return
    test = report["test"]
    spatial = report.get("headline") == "spatial" or report.get("split") == "spatial"
    qwk = test.get("quadratic_weighted_kappa")
    cards = st.columns(4, gap="medium")
    tiles = (
        ("Spatial accuracy" if spatial else "Test accuracy", f"{test['accuracy']:.3f}", "Held-out spatial cells" if spatial else "Held-out test set"),
        ("Macro F1", f"{test['f1_macro']:.3f}", "Macro average"),
        ("QWK", f"{qwk:.3f}" if qwk is not None else "—", "Ordinal agreement"),
        ("Majority baseline", f"{float(report.get('baseline_accuracy') or 0):.3f}", "Test accuracy"),
    )
    for column, (label, value, delta) in zip(cards, tiles):
        with column:
            render(kpi_card(label, value, delta))
    st.write("")
    chart_left, chart_right = st.columns(2, gap="medium")
    with chart_left:
        with st.container(border=True):
            st.plotly_chart(
                charts.class_donut(report["class_distribution"], theme),
                use_container_width=True,
                config={"displayModeBar": False},
            )
            st.caption("Labels in the processed table.")
    with chart_right:
        with st.container(border=True):
            st.plotly_chart(
                charts.confusion_heatmap(test["confusion_matrix"], test["labels"], theme),
                use_container_width=True,
                config={"displayModeBar": False},
            )
            st.caption("Held-out test set. Rows are the true class.")
    low, high = st.columns(2, gap="medium")
    with low:
        with st.container(border=True):
            from app.ui.state import importance

            payload = importance() or {}
            rows = payload.get("impurity") or []
            if rows:
                st.plotly_chart(
                    charts.importance_bars(rows, theme, "Top features", limit=5),
                    use_container_width=True,
                    config={"displayModeBar": False},
                )
            else:
                render(empty_state("No importance file", "Run python scripts/train.py to write feature_importance.json."))
            st.caption("Impurity decrease, averaged across trees. Not a cause of flooding.")
    with high:
        with st.container(border=True):
            st.plotly_chart(
                charts.metric_compare(report["train"], report["test"], theme),
                use_container_width=True,
                config={"displayModeBar": False},
            )
            gap = report.get("overfit_gap_accuracy")
            st.caption(f"Train accuracy minus test accuracy is {gap:.3f}." if gap is not None else "")
    st.write("")
    render("<h2 class='fs-section'>Recent predictions</h2>")
    history = list(st.session_state.get("history") or [])
    if not history:
        render(empty_state("No predictions yet", "Open Predict, score a location, and save it to this session."))
        return
    rows = [
        [item["when"], item["risk"], item["confidence"], str(item.get("rainfall", "—")), str(item.get("elevation", "—"))]
        for item in history[:8]
    ]
    render(data_table(["Time", "Risk", "Confidence", "Rainfall", "Elevation"], rows, badges={1: "risk"}))
