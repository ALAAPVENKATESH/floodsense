"""UI helpers and a smoke run of each FloodSense page."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from app.ui.charts import class_donut
from app.ui.logic import (
    best_depth_scores,
    driver_line,
    normalize_columns,
    ordinal,
    percentile_rank,
    prepare_batch,
    schema_errors,
    snap,
    tree_lines,
)
from app.ui.theme import chart_theme


def test_ordinal_teens() -> None:
    assert ordinal(1) == "1st"
    assert ordinal(11) == "11th"
    assert ordinal(22) == "22nd"


def test_percentile_and_driver_line() -> None:
    rank = percentile_rank([1, 2, 3, 4, 5], 1)
    assert rank == pytest.approx(20)
    text = driver_line("elevation", 4, rank)
    assert "Elevation" in text
    assert "lower than most" in text


def test_snap_lands_on_step() -> None:
    assert snap(99.15, 0, 1) == 99


def test_normalize_and_schema() -> None:
    frame = normalize_columns(pd.DataFrame({"Rain": [1], "rainfall": [2]}))
    assert list(frame.columns) == ["rainfall"]
    assert schema_errors(frame, ["rainfall", "elevation"])[0].startswith("Missing columns")


def test_prepare_batch_drops_bad_rows() -> None:
    frame = pd.DataFrame({"rainfall": [10, 5000, "x"], "elevation": [1, 1, 1]})
    valid, issues = prepare_batch(
        frame,
        ["rainfall", "elevation"],
        {"rainfall": (0, 2000), "elevation": (0, 100)},
        {"rainfall": (0, 400), "elevation": (0, 50)},
    )
    assert len(valid) == 1
    assert valid.iloc[0]["rainfall"] == 10
    assert len(issues) == 2


def test_depth_scores_keep_the_best() -> None:
    depths, scores = best_depth_scores(
        [
            {"max_depth": 8, "validation_macro_f1": 0.2},
            {"max_depth": 8, "validation_macro_f1": 0.6},
            {"max_depth": 6, "validation_macro_f1": 0.4},
        ]
    )
    assert depths == [6, 8]
    assert scores == [0.4, 0.6]


def test_tree_lines_stop_at_depth() -> None:
    leaf = SimpleNamespace(feature=None, value=[0, 2], n_samples=3, left=None, right=None)
    root = SimpleNamespace(feature=0, threshold=1.25, n_samples=6, left=leaf, right=leaf, value=[1, 1])
    lines = tree_lines(root, ["rainfall"], ["LOW", "HIGH"], max_depth=3)
    assert lines[0][1].startswith("Rainfall")
    assert any(text.startswith("Yes") for _depth, text in lines)
    assert sum(1 for _depth, text in lines if "Leaf HIGH" in text) == 2


def test_donut_is_transparent() -> None:
    fig = class_donut({"LOW": 3, "MODERATE": 2, "HIGH": 1}, chart_theme("dark"))
    assert fig.layout.paper_bgcolor == "rgba(0,0,0,0)"
    assert fig.layout.font.color == "#E5E7EB"


@pytest.mark.parametrize(
    ("page", "needle"),
    [
        ("dashboard", "Recent predictions"),
        ("predict", "Inputs"),
        ("batch", "Batch analysis"),
        ("insights", "Model insights"),
        ("dataset", "Feature dictionary"),
        ("about", "limitations"),
    ],
)
def test_each_page_runs(page: str, needle: str) -> None:
    """Each page renders once. A second AppTest run trips on an empty preset control."""
    streamlit = pytest.importorskip("streamlit.testing.v1")
    root = Path(__file__).resolve().parents[1]
    app = streamlit.AppTest.from_file(str(root / "app" / "streamlit_app.py"), default_timeout=120)
    app.query_params["page"] = page
    app.run()
    _assert_clean(app)
    text = _markdown(app)
    assert "FloodSense" in text
    assert needle in text
    if page == "dashboard":
        assert "Synthetic" in text or "SYNTHETIC" in text


def _markdown(app) -> str:
    return "\n".join(getattr(item, "value", "") for item in app.markdown)


def _assert_clean(app) -> None:
    if not app.exception:
        return
    message = "\n".join(str(getattr(item, "value", item)) for item in app.exception)
    raise AssertionError(message)
