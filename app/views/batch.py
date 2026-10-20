"""Score a CSV of locations. A map is drawn only when coordinates are present."""

from __future__ import annotations

import pandas as pd
import streamlit as st

import config
from app.ui import charts
from app.ui.components import alert, data_table, empty_state, kpi_card, render, section_header
from app.ui.logic import normalize_columns, prepare_batch, schema_errors
from app.ui.state import mode
from app.ui.theme import chart_theme
from app.views._frame import header, require_model


def page() -> None:
    """Batch analysis."""
    header("Batch analysis", "Upload a CSV of locations. Rows outside the training range are skipped.")
    fitted = require_model()
    if fitted is None:
        return
    with st.container(border=True):
        render(section_header("Upload", "Use the template headers. Latitude and longitude are optional."))
        template = _template(fitted)
        st.download_button(
            "Download template CSV",
            data=template,
            file_name="floodsense_template.csv",
            mime="text/csv",
        )
        upload = st.file_uploader("Locations CSV", type=["csv"], label_visibility="collapsed")
    if upload is None:
        if not st.session_state.get("batch_csv"):
            render(empty_state("No file yet", "Drop a CSV here, or download the template and fill one row per location."))
        else:
            _show_saved(fitted)
        return
    token = f"{upload.name}:{upload.size}"
    if st.session_state.get("batch_token") != token:
        _score(fitted, upload)
        st.session_state["batch_token"] = token
    _show_saved(fitted)


def _template(fitted) -> bytes:
    row = {feature: (fitted.extra.get("defaults") or {}).get(feature, "") for feature in fitted.base_features}
    row["latitude"] = ""
    row["longitude"] = ""
    return pd.DataFrame([row]).to_csv(index=False).encode("utf-8")


def _score(fitted, upload) -> None:
    progress = st.progress(8, text="Reading the file")
    try:
        frame = pd.read_csv(upload)
    except Exception:
        progress.empty()
        st.session_state["batch_error"] = "That file could not be read as a CSV."
        st.session_state["batch_csv"] = ""
        return
    frame = normalize_columns(frame)
    missing = schema_errors(frame, list(fitted.base_features))
    if missing:
        progress.empty()
        st.session_state["batch_error"] = " ".join(missing)
        st.session_state["batch_csv"] = ""
        return
    progress.progress(40, text="Checking values")
    valid, issues = prepare_batch(
        frame,
        list(fitted.base_features),
        config.FEATURE_RANGES,
        fitted.extra.get("input_ranges") or {},
    )
    st.session_state["batch_error"] = ""
    st.session_state["batch_issues"] = issues[:40]
    st.session_state["batch_skipped"] = len(frame) - len(valid)
    if valid.empty:
        progress.empty()
        st.session_state["batch_csv"] = ""
        st.session_state["batch_error"] = "Every row was outside the allowed ranges, so nothing was scored."
        return
    progress.progress(75, text="Scoring rows")
    predicted, probabilities, _votes = fitted.predict_frame(valid[list(fitted.base_features)])
    output = valid.copy()
    labels = list(fitted.class_labels)
    output["risk_class"] = [labels[int(index)] for index in predicted]
    for index, name in enumerate(labels):
        output[f"probability_{name}"] = probabilities[:, index]
    progress.progress(100, text="Done")
    progress.empty()
    st.session_state["batch_csv"] = output.to_csv(index=False)
    st.session_state["batch_counts"] = output["risk_class"].value_counts().to_dict()
    st.session_state["batch_preview"] = output.head(12).to_dict(orient="records")
    if {"latitude", "longitude"} <= set(output.columns):
        latitudes = pd.to_numeric(output["latitude"], errors="coerce")
        longitudes = pd.to_numeric(output["longitude"], errors="coerce")
        mask = latitudes.notna() & longitudes.notna()
        mapped = pd.DataFrame(
            {"latitude": latitudes[mask], "longitude": longitudes[mask], "risk_class": output.loc[mask, "risk_class"]}
        )
        st.session_state["batch_map"] = mapped.to_dict(orient="records") if not mapped.empty else []
    else:
        st.session_state["batch_map"] = []


def _show_saved(fitted) -> None:
    del fitted
    error = st.session_state.get("batch_error") or ""
    if error:
        render(alert(error))
    csv_text = st.session_state.get("batch_csv") or ""
    if not csv_text:
        if not error:
            render(empty_state("Nothing scored", "Check the column names against the template."))
        return
    counts = st.session_state.get("batch_counts") or {}
    cards = st.columns(4, gap="medium")
    tiles = (
        ("Rows scored", f"{sum(int(value) for value in counts.values()):,}", "Blank cells use the training median"),
        ("Low", str(int(counts.get("LOW", 0))), "Predicted class"),
        ("Moderate", str(int(counts.get("MODERATE", 0))), "Predicted class"),
        ("High", str(int(counts.get("HIGH", 0))), "Predicted class"),
    )
    for column, (label, value, delta) in zip(cards, tiles):
        with column:
            render(kpi_card(label, value, delta))
    skipped = int(st.session_state.get("batch_skipped") or 0)
    if skipped:
        render(alert(f"{skipped} row(s) were skipped. See the list under the chart."))
    theme = chart_theme(mode())
    st.plotly_chart(charts.batch_counts(counts, theme), use_container_width=True, config={"displayModeBar": False})
    preview = st.session_state.get("batch_preview") or []
    if preview:
        columns = ["risk_class"] + [name for name in preview[0] if name != "risk_class"]
        show = columns[:6]
        rows = [[str(item.get(name, "")) for name in show] for item in preview]
        risk_index = show.index("risk_class")
        render(data_table([name.replace("_", " ") for name in show], rows, badges={risk_index: "risk"}))
        st.caption("First 12 scored rows. The download has every scored row and the class probabilities.")
    mapped = st.session_state.get("batch_map") or []
    if mapped:
        st.plotly_chart(
            charts.location_map(pd.DataFrame(mapped), theme),
            use_container_width=True,
            config={"displayModeBar": False},
        )
        st.caption("Drawn only because this file has latitude and longitude.")
    issues = st.session_state.get("batch_issues") or []
    if issues:
        render(section_header("Skipped rows", "These values were not clipped into a prediction."))
        render(data_table(["CSV row", "Issue"], [[str(item["row"]), item["issue"]] for item in issues]))
    st.download_button(
        "Download predictions CSV",
        data=csv_text,
        file_name="floodsense_predictions.csv",
        mime="text/csv",
        type="primary",
    )
