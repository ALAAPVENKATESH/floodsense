"""What the model was trained on, including the synthetic-data warning."""

from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

import config
from app.ui import charts
from app.ui.components import callout, data_table, empty_state, render, section_header
from app.ui.state import dataset_meta, mode, model, table
from app.ui.theme import UNITS, chart_theme, display_name
from app.views._frame import header
from flood_rf.features import FORMULAS, FeatureEngineer


def page() -> None:
    """Dataset page."""
    report = header("Dataset", "Where the rows came from, what is missing, and how the columns relate.")
    frame = table()
    if frame is None:
        render(empty_state("No processed table", "Run python scripts/build_dataset.py to write data/processed/dataset.csv."))
        return
    fitted = model()
    meta = dataset_meta()
    source = str((report or meta or {}).get("data_source") or "UNKNOWN")
    notice = config.warning_for(source)
    if notice:
        render(callout(notice))
    _overview(frame, report, meta, source)
    engineered = FeatureEngineer().transform(frame)
    _dictionary(engineered, fitted)
    _missing(frame)
    theme = chart_theme(mode())
    _charts(engineered, report, theme)


def _overview(frame: pd.DataFrame, report: dict | None, meta: dict, source: str) -> None:
    rows = len(frame)
    columns = len(frame.columns)
    if source == "SYNTHETIC":
        coverage = (
            f"No surveyed region or time period. Generator seed {meta.get('generator_seed', config.RANDOM_STATE)}. "
            "Labels are a noisy binning of a latent score, not a flood agency class."
        )
    elif source == config.BENCHMARK_SOURCE:
        coverage = meta.get("coverage") or config.BENCHMARK_COVERAGE
    else:
        coverage = meta.get("coverage") or (
            "Add geographic coverage, time period, dataset version, and license in docs/dataset_documentation.md."
        )
    rule = (report or {}).get("label_rule") or meta.get("label_rule") or "See docs/dataset_documentation.md."
    render(section_header("Coverage", coverage))
    render(data_table(
        ["Item", "Value"],
        [
            ["Source", source],
            ["Rows", f"{rows:,}"],
            ["Columns", str(columns)],
            ["Train / test", _split(report)],
            ["Label rule", rule],
            ["Documentation", "docs/dataset_documentation.md"],
        ],
    ))


def _sources() -> None:
    path = config.DATA_PROCESSED / "external_meta.json"
    if not path.exists():
        return
    import json

    meta = json.loads(path.read_text(encoding="utf-8"))
    render(section_header("External sources", "Joined on latitude, longitude, and year. A missing pixel stays missing."))
    chirps = meta.get("chirps") or {}
    population = meta.get("population") or {}
    corr = meta.get("file_vs_chirps") or {}
    pearson = corr.get("pearson")
    pearson_text = "—" if pearson is None else f"{float(pearson):.3f}"
    render(data_table(
        ["Source", "Detail"],
        [
            ["CHIRPS", f"{chirps.get('dataset', 'CHIRPS v2.0')}, {chirps.get('resolution', '0.05°')}, public domain"],
            ["CHIRPS window", f"{chirps.get('window_deg', 0.25)}° neighbourhood mean"],
            ["CHIRPS baseline", chirps.get("climatology", "1981–2010")],
            ["Population", population.get("primary", "WorldPop")],
            ["Population fallback", population.get("fallback", "GPW v4")],
            ["Radius sum", population.get("radius", "")],
            ["File rainfall vs CHIRPS", f"Pearson {pearson_text} on {corr.get('n', 0)} rows"],
        ],
    ))
    decades = meta.get("decades") or []
    if decades:
        render(data_table(
            ["Decade", "Rows", "CHIRPS missing", "Population missing", "Year clipped"],
            [
                [
                    str(row["decade"]),
                    str(row["n"]),
                    f"{float(row['rain_missing']):.0%}",
                    f"{float(row['pop_missing']):.0%}",
                    f"{float(row['pop_year_clipped']):.0%}",
                ]
                for row in decades
            ],
        ))


def _split(report: dict | None) -> str:
    if not report:
        return "Run train.py to record the split."
    return f"{report['n_train']:,} / {report['n_test']:,}"


def _dictionary(frame: pd.DataFrame, fitted) -> None:
    ranges = (fitted.extra.get("input_ranges") if fitted is not None else {}) or {}
    rows = []
    for name in frame.columns:
        if name in {config.TARGET, config.SOURCE_COLUMN, *config.LABEL_SCORE_COLUMNS}:
            continue
        description = config.FEATURE_HELP.get(name) or FORMULAS.get(name) or "Identifier or source field."
        unit = UNITS.get(name, "—")
        if name in ranges:
            low, high = ranges[name]
            span = f"{float(low):.4g} – {float(high):.4g}"
        elif name in config.FEATURE_RANGES:
            low, high = config.FEATURE_RANGES[name]
            span = f"{float(low):.4g} – {float(high):.4g}"
        else:
            span = "—"
        rows.append([display_name(name), description, unit, span])
    render(section_header("Feature dictionary", "Ranges shown for model inputs are the training min and max."))
    render(data_table(["Feature", "Description", "Unit", "Range"], rows))


def _missing(frame: pd.DataFrame) -> None:
    share = frame.isna().mean().sort_values(ascending=False)
    share = share[share > 0]
    render(section_header("Missing values", "Share of blank cells. The imputer fills these from the training split only."))
    if share.empty:
        render(empty_state("No blank cells", "Every column in the processed table is filled."))
        return
    render(data_table(
        ["Column", "Missing"],
        [[display_name(str(name)), f"{float(value):.1%}"] for name, value in share.items()],
    ))


def _charts(frame: pd.DataFrame, report: dict | None, theme) -> None:
    if config.TARGET in frame.columns:
        counts = frame[config.TARGET].value_counts().to_dict()
    elif report:
        counts = report.get("class_distribution") or {}
    else:
        counts = {}
    if counts:
        st.plotly_chart(charts.class_donut(counts, theme), use_container_width=True, config={"displayModeBar": False})
    numeric = frame.select_dtypes(include="number")
    id_only = [name for name in config.ID_COLUMNS if name not in config.BASE_FEATURES + config.OPTIONAL_FEATURES]
    blocked = set(id_only) | {config.TARGET} | set(config.LABEL_SCORE_COLUMNS) | set(config.INDICATOR_FEATURES)
    drop = [name for name in numeric.columns if name in blocked]
    numeric = numeric.drop(columns=drop, errors="ignore")
    numeric = numeric.loc[:, numeric.notna().any()]
    if numeric.shape[1] >= 2:
        corr = numeric.corr().to_numpy()
        if np.isfinite(corr).any():
            st.plotly_chart(
                charts.correlation_heatmap(list(numeric.columns), corr, theme),
                use_container_width=True,
                config={"displayModeBar": False},
            )
            st.caption("Pearson correlation on the processed table, after the documented feature formulas.")
    if numeric.shape[1] == 0:
        return
    feature = st.selectbox("Distribution", list(numeric.columns), format_func=display_name)
    values = numeric[feature].dropna().to_numpy()
    log_scale = feature in set(config.POP_EXT) | {"rain_annual_mm", "rain_max_month_mm"}
    st.plotly_chart(
        charts.histogram(values, feature, theme, log_scale=log_scale),
        use_container_width=True,
        config={"displayModeBar": False},
    )
    _sources()
