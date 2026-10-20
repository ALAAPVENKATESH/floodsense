"""Holdout performance, importance, and a read-only look at one tree."""

from __future__ import annotations

from html import escape

import streamlit as st

import config
from app.ui import charts
from app.ui.components import callout, data_table, empty_state, render, section_header, tree_block
from app.ui.logic import best_depth_scores, tree_lines
from app.ui.state import importance, mode
from app.ui.theme import chart_theme
from app.views._frame import header, require_model


def page() -> None:
    """Model insights."""
    report = header("Model insights", "Held-out scores, what the forest used, and one example tree.")
    fitted = require_model()
    if fitted is None or report is None:
        if fitted is not None and report is None:
            render(empty_state("Metrics missing", "Run python scripts/train.py."))
        return
    theme = chart_theme(mode())
    performance, features, forest = st.tabs(["Performance", "Feature importance", "Forest explorer"])
    with performance:
        _performance(report, theme)
    with features:
        _features(theme)
    with forest:
        _forest(fitted, report, theme)


def _performance(report: dict, theme) -> None:
    test = report["test"]
    render(section_header("Per-class scores", "Darker cells are higher. Support is the test-set count."))
    render(_shaded(test["per_class"], theme.mode))
    choice = st.radio("Matrix", ["Counts", "Normalized"], horizontal=True, key="matrix_mode")
    st.plotly_chart(
        charts.confusion_heatmap(
            test["confusion_matrix"],
            test["labels"],
            theme,
            normalized=choice == "Normalized",
        ),
        use_container_width=True,
        config={"displayModeBar": False},
    )
    st.plotly_chart(
        charts.metric_compare(report["train"], report["test"], theme),
        use_container_width=True,
        config={"displayModeBar": False},
    )
    gap = float(report.get("overfit_gap_accuracy") or 0)
    baseline = float(report.get("baseline_accuracy") or 0)
    oob = report.get("oob_score")
    oob_text = f"{float(oob):.3f}" if oob is not None else "n/a"
    def _num(key: str) -> str:
        value = test.get(key)
        return f"{float(value):.3f}" if isinstance(value, (int, float)) else "n/a"

    split_name = report.get("headline") or report.get("split") or "test"
    st.caption(
        f"{split_name} accuracy {test['accuracy']:.3f}. Macro F1 {test['f1_macro']:.3f}. "
        f"QWK {_num('quadratic_weighted_kappa')}. Adjacent accuracy {_num('adjacent_accuracy')}. "
        f"Mean absolute class error {_num('mean_absolute_class_error')}. "
        f"Severe error rate {_num('severe_error_rate')}. "
        f"Train accuracy {report['train']['accuracy']:.3f}. Gap {gap:.3f}. "
        f"Majority baseline {baseline:.3f}. Out-of-bag accuracy {oob_text}."
    )
    _comparisons(report)


def _features(theme) -> None:
    payload = importance()
    if not payload:
        render(empty_state("No importance file", "Run python scripts/train.py."))
        return
    render(callout(payload.get("note") or config.CAUSATION_NOTE))
    left, right = st.columns(2, gap="medium")
    with left:
        st.plotly_chart(
            charts.importance_bars(payload.get("impurity") or [], theme, "Impurity importance"),
            use_container_width=True,
            config={"displayModeBar": False},
        )
    with right:
        permutation = payload.get("permutation") or []
        if permutation:
            st.plotly_chart(
                charts.importance_bars(
                    permutation,
                    theme,
                    "Permutation importance",
                    x_title="Accuracy drop when shuffled",
                ),
                use_container_width=True,
                config={"displayModeBar": False},
            )
            st.caption("Held-out shuffle. It was not used to choose the model.")
        else:
            render(empty_state("No permutation scores", "The importance file has impurity scores only."))


def _forest(fitted, report: dict, theme) -> None:
    curve = report.get("learning_curve") or []
    if curve:
        st.plotly_chart(charts.learning_curve(curve, theme), use_container_width=True, config={"displayModeBar": False})
        st.caption("Out-of-bag accuracy on the training rows as trees are added.")
    depths, scores = best_depth_scores(report.get("search") or [])
    if depths:
        st.plotly_chart(charts.depth_curve(depths, scores, theme), use_container_width=True, config={"displayModeBar": False})
        st.caption("Best validation macro-F1 at each depth in the search log. Other settings also changed.")
    trees = fitted.forest.trees_
    if not trees:
        render(empty_state("No trees", "The saved forest has no trees."))
        return
    render(section_header("Example tree", "Read-only. Top 3 levels. A leaf is the majority class in that node."))
    index = st.selectbox(
        "Tree",
        options=list(range(len(trees))),
        format_func=lambda number: f"Tree {number + 1} of {len(trees)}",
    )
    lines = tree_lines(trees[index].root_, fitted.feature_names, fitted.class_labels, max_depth=3)
    render(tree_block(lines))


def _comparisons(report: dict) -> None:
    rows = [["Spatial test" if report.get("headline") == "spatial" else "Test", _cell(report.get("test"), "accuracy"), _cell(report.get("test"), "f1_macro"), _cell(report.get("test"), "quadratic_weighted_kappa")]]
    random_split = report.get("random_split") or {}
    if random_split:
        rows.append(["Random split (optimistic, leakage-prone)", _cell(random_split, "accuracy"), _cell(random_split, "f1_macro"), _cell(random_split, "quadratic_weighted_kappa")])
    baselines = report.get("baselines") or {}
    if baselines.get("majority"):
        rows.append(["Majority class", _cell(baselines["majority"], "accuracy"), _cell(baselines["majority"], "f1_macro"), _cell(baselines["majority"], "quadratic_weighted_kappa")])
    if baselines.get("stratified_random_guess"):
        rows.append(["Stratified random guess", _cell(baselines["stratified_random_guess"], "accuracy"), _cell(baselines["stratified_random_guess"], "f1_macro"), _cell(baselines["stratified_random_guess"], "quadratic_weighted_kappa")])
    if len(rows) > 1:
        render(section_header("Splits and baselines", "The spatial row is the headline. The random row can look better because nearby points leak."))
        render(data_table(["Comparison", "Accuracy", "Macro F1", "QWK"], rows))
    ablation = report.get("ablation") or []
    if ablation:
        render(section_header("Ablation", "Same spatial split and the same chosen settings. A higher score is not proof that the feature causes flooding."))
        render(data_table(
            ["Features", "Accuracy", "Macro F1", "QWK"],
            [[row.get("name", ""), _cell(row, "accuracy"), _cell(row, "f1_macro"), _cell(row, "quadratic_weighted_kappa")] for row in ablation],
        ))
    variance = report.get("seed_variance") or {}
    if variance:
        render(section_header("Five seeds", "Same settings, different spatial splits. Mean ± sample standard deviation."))
        render(data_table(
            ["Metric", "Mean", "Std"],
            [[name, f"{float(stats['mean']):.3f}", f"{float(stats['std']):.3f}"] for name, stats in variance.items()],
        ))
    _external(report)


def _external(report: dict) -> None:
    external = report.get("external") or {}
    means = external.get("ablation_mean") or []
    if means:
        render(section_header("External ablation", "Same cells inside each population. Mean ± std across spatial-split seeds."))
        render(data_table(
            ["Population", "Features", "Accuracy", "Macro F1", "QWK"],
            [
                [
                    str(row.get("population", "")),
                    str(row.get("feature_set", "")),
                    _pm(row, "accuracy"),
                    _pm(row, "f1_macro"),
                    _pm(row, "quadratic_weighted_kappa"),
                ]
                for row in means
            ],
        ))
    paired = external.get("paired_A") or {}
    per_seed = paired.get("per_seed") or []
    if per_seed:
        render(section_header("Paired seeds", "Full model minus the latitude/longitude reference, on population A."))
        render(data_table(
            ["Seed", "Accuracy diff", "Macro F1 diff", "QWK diff"],
            [
                [
                    str(row.get("seed")),
                    f"{float(row.get('full_minus_latlon_accuracy', 0)):.3f}",
                    f"{float(row.get('full_minus_latlon_f1_macro', 0)):.3f}",
                    f"{float(row.get('full_minus_latlon_quadratic_weighted_kappa', 0)):.3f}",
                ]
                for row in per_seed
            ],
        ))
    if external.get("cell_15"):
        cell = external["cell_15"]
        drop = external.get("cell_15_accuracy_drop")
        render(section_header("15° cells", "Same features, coarser cells. The held-out regions change."))
        render(data_table(
            ["Accuracy", "Macro F1", "QWK", "Accuracy drop"],
            [[_cell(cell, "accuracy"), _cell(cell, "f1_macro"), _cell(cell, "quadratic_weighted_kappa"), "—" if drop is None else f"{float(drop):.3f}"]],
        ))
    continents = [row for row in (external.get("continents") or []) if "accuracy" in row]
    if continents:
        render(section_header("Leave-one-continent-out", "Boxes are documented in the preprocessing module, not a coastline."))
        render(data_table(
            ["Continent", "Test rows", "Accuracy", "Macro F1", "QWK"],
            [
                [row["continent"], str(row["n_test"]), f"{float(row['accuracy']):.3f}", f"{float(row['f1_macro']):.3f}", f"{float(row['quadratic_weighted_kappa']):.3f}"]
                for row in continents
            ],
        ))


def _pm(row: dict, key: str) -> str:
    if row.get(key) is None:
        return "—"
    std = row.get(f"{key}_std")
    text = f"{float(row[key]):.3f}"
    if std is not None:
        text += f" ± {float(std):.3f}"
    return text


def _cell(payload: dict | None, key: str) -> str:
    if not payload or payload.get(key) is None:
        return "—"
    return f"{float(payload[key]):.3f}"


def _shaded(per_class: dict, mode_name: str) -> str:
    ink = "#E5E7EB" if mode_name == "dark" else "#0F172A"
    rows = []
    for label, scores in per_class.items():
        cells = [f"<td>{escape(str(label))}</td>"]
        for key in ("precision", "recall", "f1"):
            value = float(scores[key])
            alpha = 0.12 + 0.5 * value
            cells.append(
                f"<td style='background:rgba(79,70,229,{alpha:.2f});color:{ink}'>{value:.3f}</td>"
            )
        cells.append(f"<td>{int(scores['support'])}</td>")
        rows.append("<tr>" + "".join(cells) + "</tr>")
    head = "".join(f"<th>{name}</th>" for name in ("Class", "Precision", "Recall", "F1", "Support"))
    return f"<table class='fs-table'><thead><tr>{head}</tr></thead><tbody>{''.join(rows)}</tbody></table>"
