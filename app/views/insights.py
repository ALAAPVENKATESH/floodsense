"""Holdout performance, importance, and a read-only look at one tree."""

from __future__ import annotations

from html import escape

import streamlit as st

import config
from app.ui import charts
from app.ui.components import callout, empty_state, render, section_header, tree_block
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
    st.caption(
        f"Train accuracy {report['train']['accuracy']:.3f} versus test {test['accuracy']:.3f}. "
        f"Gap {gap:.3f}. Majority-class baseline {baseline:.3f}. Out-of-bag accuracy {oob_text}."
    )


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
