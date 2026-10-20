"""Single-location scoring, vote breakdown, and a one-feature what-if."""

from __future__ import annotations

import json
import subprocess
import sys
from datetime import datetime

import numpy as np
import pandas as pd
import streamlit as st

import config
from app.ui import charts
from app.ui.components import alert, callout, empty_state, render, risk_result_card, section_header
from app.ui.logic import driver_line, input_step, percentile_rank, slider_bounds, snap
from app.ui.state import mode, training_rows
from app.ui.theme import GROUPS, UNITS, chart_theme, display_name
from app.views._frame import header, require_model

_CAVEAT = "Prediction reflects model behavior on training data, not a guarantee."
_EXPOSURE = (
    "Predictions depend on location and exposure data; "
    "weak sensitivity to one input does not mean the model is broken."
)


def page() -> None:
    """Predict page."""
    header("Predict", "Set rainfall, terrain, drainage, and history, then score one location.")
    fitted = require_model()
    if fitted is None:
        render(callout(_CAVEAT))
        return
    _ensure_inputs(fitted)
    render(callout(_EXPOSURE))
    record = {feature: float(st.session_state[feature]) for feature in fitted.base_features}
    errors = fitted.validate(record)
    left, right = st.columns([1.05, 0.95], gap="medium")
    with left:
        _inputs(fitted, record)
    with right:
        _result(fitted, record, errors)
    st.write("")
    _what_if(fitted, record, errors)
    render(callout(_CAVEAT))


def _ensure_inputs(fitted) -> None:
    """Apply a preset or the reset before any input widget is created."""
    if st.session_state.pop("_reset", False):
        _write(fitted, fitted.extra.get("defaults") or {})
        st.session_state["preset_choice"] = None
        st.session_state["_applied_preset"] = None
    else:
        choice = st.session_state.get("preset_choice")
        if choice and choice != st.session_state.get("_applied_preset"):
            preset = (fitted.extra.get("presets") or {}).get(str(choice).upper())
            if preset:
                _write(fitted, preset)
            st.session_state["_applied_preset"] = choice
    looked = st.session_state.pop("_lookup_values", None)
    if looked:
        _write(fitted, looked)
    defaults = fitted.extra.get("defaults") or {}
    for feature in fitted.base_features:
        if feature in st.session_state and f"{feature}__s" in st.session_state:
            continue
        _write(fitted, {feature: defaults.get(feature, _bounds(fitted, feature)[0])})


def _write(fitted, values: dict) -> None:
    for feature, raw in values.items():
        if feature not in fitted.base_features or raw is None:
            continue
        low, high = _widget_bounds(feature)
        value = min(max(snap(float(raw), low, input_step(feature)), low), high)
        st.session_state[feature] = value
        st.session_state[f"{feature}__s"] = value


def _bounds(fitted, feature: str) -> tuple[float, float]:
    ranges = fitted.extra.get("input_ranges") or {}
    if feature in ranges:
        low, high = ranges[feature]
        return float(low), float(high)
    return _physical(feature)


def _physical(feature: str) -> tuple[float, float]:
    if feature in config.FEATURE_RANGES:
        low, high = config.FEATURE_RANGES[feature]
        return float(low), float(high)
    return 0.0, 1.0


def _widget_bounds(feature: str) -> tuple[float, float]:
    low, high = _physical(feature)
    return slider_bounds(low, high, input_step(feature))


def _from_slider(feature: str) -> None:
    st.session_state[feature] = st.session_state[f"{feature}__s"]


def _from_number(feature: str, low: float, high: float) -> None:
    value = float(st.session_state[feature])
    st.session_state[f"{feature}__s"] = min(max(value, low), high)


def _inputs(fitted, record: dict) -> None:
    with st.container(border=True):
        render(section_header("Inputs", "Presets are training-set medians for that class."))
        st.segmented_control(
            "Load preset",
            options=["Low", "Moderate", "High"],
            key="preset_choice",
            selection_mode="single",
        )
        train_ranges = fitted.extra.get("input_ranges") or {}
        present = set(fitted.base_features)
        for title, features in GROUPS:
            shown = [feature for feature in features if feature in present]
            if not shown:
                continue
            render(f"<div class='fs-group'>{title}</div>")
            for feature in shown:
                step = input_step(feature)
                unit = UNITS.get(feature, "")
                label = display_name(feature) + (f" ({unit})" if unit else "")
                train = train_ranges.get(feature)
                phys_low, phys_high = _widget_bounds(feature)
                if train:
                    slide_low, slide_high = slider_bounds(float(train[0]), float(train[1]), step)
                else:
                    slide_low, slide_high = phys_low, phys_high
                if slide_high <= slide_low:
                    slide_high = round(slide_low + step, 6)
                current_slide = float(st.session_state[f"{feature}__s"])
                if current_slide < slide_low or current_slide > slide_high:
                    st.session_state[f"{feature}__s"] = min(max(current_slide, slide_low), slide_high)
                help_text = config.FEATURE_HELP.get(feature, feature)
                if train:
                    help_text += f" Training range {float(train[0]):.4g} to {float(train[1]):.4g}."
                number, slider = st.columns([1, 1.3])
                with slider:
                    st.slider(
                        label,
                        min_value=slide_low,
                        max_value=slide_high,
                        step=step,
                        key=f"{feature}__s",
                        label_visibility="collapsed",
                        on_change=_from_slider,
                        args=(feature,),
                        help=help_text,
                    )
                with number:
                    st.number_input(
                        label,
                        min_value=phys_low,
                        max_value=phys_high,
                        step=step,
                        key=feature,
                        on_change=_from_number,
                        args=(feature, slide_low, slide_high),
                        help=help_text,
                    )
                if train and (record[feature] < float(train[0]) or record[feature] > float(train[1])):
                    if feature in {"latitude", "longitude", "year"}:
                        message = (
                            f"{display_name(feature)} is outside the training coverage "
                            f"{float(train[0]):.4g}–{float(train[1]):.4g}. Prediction refused."
                        )
                    else:
                        message = (
                            f"{display_name(feature)} is outside the training range "
                            f"{float(train[0]):.4g}–{float(train[1]):.4g}. Prediction refused."
                        )
                    render(alert(message))
        year = _lookup_year(fitted, record)
        _coverage(record, year)
        _lookup(fitted, record, year)
        reset, _spacer = st.columns([1, 2])
        with reset:
            if st.button("Reset", use_container_width=True):
                st.session_state["_reset"] = True
                st.rerun()


def _lookup_year(fitted, record: dict) -> float | None:
    """Year is not a saved-model input. The satellite lookup still needs one."""
    if "year" in record:
        return float(record["year"])
    if not {"latitude", "longitude"} <= set(fitted.base_features) or not _rasters_present():
        return None
    if "lookup_year" not in st.session_state:
        st.session_state["lookup_year"] = 2010
    return float(
        st.number_input("Year for the satellite lookup", min_value=1900, max_value=2020, step=1, key="lookup_year")
    )


def _coverage(record: dict, year: float | None) -> None:
    lat = record.get("latitude")
    if year is not None and year < 1981:
        render(alert("CHIRPS rainfall starts in 1981. This year has no satellite rainfall."))
    if year is not None and (year < 2000 or year > 2020):
        render(alert("Population is clipped to the WorldPop range 2000–2020."))
    if lat is not None and (lat < -50 or lat > 50):
        render(alert("CHIRPS covers 50°S to 50°N. This latitude is outside that grid."))


def _rasters_present() -> bool:
    root = config.ROOT / "data" / "external"
    chirps = root / "chirps"
    pop = root / "worldpop"
    has_chirps = chirps.exists() and any(chirps.glob("chirps-v2.0.*.tif.gz"))
    has_pop = pop.exists() and any(pop.glob("ppp_*_1km_Aggregated.tif"))
    return has_chirps or has_pop


def _apply_lookup(lat: float, lon: float, year: float) -> None:
    script = config.ROOT / "scripts" / "build_external_features.py"
    proc = subprocess.run(
        [sys.executable, str(script), "--lookup", str(lat), str(lon), str(int(year))],
        capture_output=True,
        text=True,
        timeout=180,
        cwd=str(config.ROOT),
    )
    if proc.returncode != 0 or not proc.stdout.strip():
        st.session_state["_lookup_error"] = (proc.stderr or proc.stdout or "lookup failed")[-400:]
        return
    st.session_state["_lookup_values"] = json.loads(proc.stdout)
    st.session_state["_lookup_error"] = ""


def _lookup(fitted, record: dict, year: float | None) -> None:
    if not {"latitude", "longitude"} <= set(fitted.base_features):
        return
    if not _rasters_present():
        st.caption("Look up from satellite data is hidden until CHIRPS or WorldPop files are in data/external/.")
        return
    if year is None:
        return
    st.button(
        "Look up from satellite data",
        on_click=_apply_lookup,
        args=(record["latitude"], record["longitude"], year),
    )
    error = st.session_state.get("_lookup_error")
    if error:
        render(alert(error))


def _result(fitted, record: dict, errors: list[str]) -> None:
    with st.container(border=True):
        render(section_header("Result", "Majority vote of the fitted trees."))
        if errors:
            for message in errors:
                render(alert(message))
            render(empty_state("Prediction refused", "Move the highlighted inputs into the training range."))
            return
        scored = st.session_state.get("pred_inputs")
        result = st.session_state.get("pred_result")
        fresh = scored == record and result is not None
        if fresh:
            _draw_result(fitted, record, result)
        else:
            render(empty_state("Ready to score", "Choose Predict to classify these conditions."))
        predict, save, download = st.columns(3)
        with predict:
            if st.button("Predict", type="primary", use_container_width=True):
                try:
                    st.session_state["pred_result"] = fitted.predict_record(record)
                    st.session_state["pred_inputs"] = dict(record)
                    st.rerun()
                except ValueError as exc:
                    render(alert(str(exc)))
        with save:
            if st.button("Save to history", use_container_width=True, disabled=not fresh):
                _save(record, result)
                st.toast("Saved to this session")
        with download:
            payload = _payload(record, result if fresh else None)
            st.download_button(
                "Download JSON",
                data=json.dumps(payload, indent=2),
                file_name="floodsense_prediction.json",
                mime="application/json",
                use_container_width=True,
                disabled=not fresh,
            )


def _draw_result(fitted, record: dict, result: dict) -> None:
    label = result["risk_class"]
    confidence = 100.0 * float(result["probabilities"][label])
    vote_text = f"{result['trees_for_predicted']} of {result['n_trees']} trees voted {label}"
    render(risk_result_card(label, confidence, vote_text))
    theme = chart_theme(mode())
    st.plotly_chart(charts.gauge(confidence, label, theme), use_container_width=True, config={"displayModeBar": False})
    st.plotly_chart(
        charts.probability_bars(result["probabilities"], theme),
        use_container_width=True,
        config={"displayModeBar": False},
    )
    st.plotly_chart(charts.vote_bar(result["votes"], theme), use_container_width=True, config={"displayModeBar": False})
    render(section_header("Key drivers", "Where this input sits in the training rows."))
    _drivers(fitted, record)


def _drivers(fitted, record: dict) -> None:
    ranks = list(zip(fitted.feature_names, fitted.forest.feature_importances_))
    ordered = [name for name, _score in sorted(ranks, key=lambda item: -float(item[1])) if name in record]
    train = training_rows()
    if train is None:
        render(empty_state("No training table", "Percentiles need data/processed/dataset.csv."))
        return
    lines = []
    for feature in ordered[:5]:
        if feature not in train.columns:
            continue
        percent = percentile_rank(train[feature].to_numpy(), record[feature])
        lines.append(driver_line(feature, record[feature], percent))
    if not lines:
        return
    items = "".join(f"<li>{line}</li>" for line in lines)
    render(f"<ul class='fs-section-note'>{items}</ul>")


def _save(record: dict, result: dict) -> None:
    label = result["risk_class"]
    item = {
        "when": datetime.now().strftime("%H:%M:%S"),
        "risk": label,
        "confidence": f"{100.0 * float(result['probabilities'][label]):.0f}%",
        "rainfall": record.get("rainfall", "—"),
        "elevation": record.get("elevation", "—"),
    }
    history = list(st.session_state.get("history") or [])
    history.insert(0, item)
    st.session_state["history"] = history[:12]


def _payload(record: dict, result: dict | None) -> dict:
    return {
        "product": "FloodSense",
        "inputs": record,
        "prediction": result,
        "caveat": _CAVEAT,
    }


def _what_if(fitted, record: dict, errors: list[str]) -> None:
    with st.container(border=True):
        render(section_header("What-if", "Sweep one input across its training range. The other inputs stay fixed."))
        if errors:
            render(alert("Fix the range warnings above before the sweep."))
            return
        ranks = list(zip(fitted.feature_names, fitted.forest.feature_importances_))
        options = [name for name, _score in sorted(ranks, key=lambda item: -float(item[1])) if name in record]
        if not options:
            return
        feature = st.selectbox("Feature to sweep", options, format_func=display_name)
        low, high = _bounds(fitted, feature)
        if high <= low:
            render(empty_state("No range to sweep", "This input is constant in the training data."))
            return
        grid = np.linspace(low, high, 21)
        rows = []
        for value in grid:
            row = dict(record)
            row[feature] = float(value)
            rows.append(row)
        _predicted, probabilities, _votes = fitted.predict_frame(pd.DataFrame(rows))
        high_index = fitted.class_labels.index("HIGH")
        st.plotly_chart(
            charts.what_if(grid, probabilities[:, high_index], record[feature], feature, chart_theme(mode())),
            use_container_width=True,
            config={"displayModeBar": False},
        )
        st.caption("The dotted line is the current value. The curve is P(HIGH), not a physical forecast.")
