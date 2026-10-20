"""Pure helpers for the FloodSense UI. No Streamlit calls."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

import config
from app.ui.theme import STEPS, display_name


def snap(value: float, low: float, step: float) -> float:
    """Round ``value`` onto the slider grid that starts at ``low``."""
    if step <= 0:
        return float(value)
    steps = round((float(value) - float(low)) / float(step))
    return round(float(low) + steps * float(step), 6)


def slider_bounds(low: float, high: float, step: float) -> tuple[float, float]:
    """Inclusive bounds that land on ``step`` so Streamlit accepts the widget."""
    low = float(low)
    high = float(high)
    if high < low:
        low, high = high, low
    if step <= 0 or high == low:
        return low, high
    count = math.floor((high - low) / step + 1e-9)
    end = round(low + count * step, 6)
    if end < low:
        end = low
    return low, end


def ordinal(number: int) -> str:
    """1 -> 1st, 11 -> 11th, 22 -> 22nd."""
    value = int(number)
    if 10 <= (value % 100) <= 13:
        suffix = "th"
    else:
        suffix = {1: "st", 2: "nd", 3: "rd"}.get(value % 10, "th")
    return f"{value}{suffix}"


def percentile_rank(values, point: float) -> float:
    """Percent of finite training values that are less than or equal to ``point``."""
    array = np.asarray(values, dtype=np.float64)
    finite = array[np.isfinite(array)]
    if finite.size == 0:
        return 50.0
    return float(np.mean(finite <= float(point)) * 100.0)


def driver_line(feature: str, value: float, percent: float) -> str:
    """One key-driver sentence: value, percentile, and where it sits in training."""
    unit = {
        "rainfall": "mm",
        "rainfall_intensity": "mm/h",
        "rainfall_antecedent": "mm",
        "elevation": "m",
        "slope": "°",
        "drainage_density": "km/km²",
        "impervious_area": "%",
        "distance_to_water": "km",
        "historical_flood": "",
    }.get(feature, "")
    if percent < 40:
        relation = "lower than most training locations"
    elif percent > 60:
        relation = "higher than most training locations"
    else:
        relation = "near the middle of the training locations"
    unit_bit = f" {unit}" if unit else ""
    rank = ordinal(int(round(percent)))
    return (
        f"{display_name(feature)}: {float(value):.4g}{unit_bit} · "
        f"{rank} percentile — {relation}"
    )


def normalize_columns(frame: pd.DataFrame) -> pd.DataFrame:
    """Map header aliases from config onto the model's column names."""
    mapping = {}
    for column in frame.columns:
        key = str(column).strip().lower().replace(" ", "_")
        mapping[column] = config.COLUMN_ALIASES.get(key, key)
    renamed = frame.rename(columns=mapping)
    return renamed.loc[:, ~renamed.columns.duplicated()]


def schema_errors(frame: pd.DataFrame, required: list[str]) -> list[str]:
    """Friendly messages when a batch file is missing model inputs."""
    missing = [name for name in required if name not in frame.columns]
    if not missing:
        return []
    joined = ", ".join(missing)
    return [
        f"Missing columns: {joined}. Download the template and keep those headers."
    ]


def prepare_batch(
    frame: pd.DataFrame,
    features: list[str],
    physical: dict,
    training: dict,
) -> tuple[pd.DataFrame, list[dict]]:
    """Keep rows the model can score. Out-of-range rows are listed, not clipped.

    Blank cells stay blank so the saved imputer can fill them. A non-numeric
    cell, or a number outside the physical or training range, drops the row.
    """
    work = frame.copy()
    keep = np.ones(len(work), dtype=bool)
    issues: list[dict] = []
    for feature in features:
        numeric = pd.to_numeric(work[feature], errors="coerce")
        raw = work[feature]
        bad_type = raw.notna().to_numpy() & numeric.isna().to_numpy()
        for position in np.flatnonzero(bad_type):
            issues.append({"row": int(position) + 2, "issue": f"{feature} is not a number"})
        outside = np.zeros(len(work), dtype=bool)
        values = numeric.to_numpy(dtype=np.float64)
        finite = np.isfinite(values)
        for bounds, label in ((physical, "allowed"), (training, "training")):
            if feature not in bounds:
                continue
            low, high = bounds[feature]
            hit = finite & ((values < low) | (values > high))
            for position in np.flatnonzero(hit & ~outside & ~bad_type):
                issues.append(
                    {
                        "row": int(position) + 2,
                        "issue": (
                            f"{feature}={values[position]:.4g} is outside the {label} "
                            f"range [{low:.4g}, {high:.4g}]"
                        ),
                    }
                )
            outside |= hit
        keep &= ~bad_type & ~outside
        work[feature] = numeric
    valid = work.loc[keep].copy()
    return valid, issues


def best_depth_scores(search: list[dict]) -> tuple[list[int], list[float]]:
    """Best validation macro-F1 seen at each max_depth in the search log."""
    best: dict[int, float] = {}
    for row in search:
        depth = int(row["max_depth"])
        score = float(row["validation_macro_f1"])
        best[depth] = max(score, best.get(depth, 0.0))
    depths = sorted(best)
    return depths, [best[depth] for depth in depths]


def tree_lines(root, feature_names: list[str], class_labels: list[str], max_depth: int = 3) -> list[tuple[int, str]]:
    """Indented text for the top levels of one fitted tree. Read only."""
    lines: list[tuple[int, str]] = []

    def walk(node, depth: int, edge: str) -> None:
        if node is None or depth > max_depth:
            return
        if getattr(node, "feature", None) is None:
            counts = np.asarray(node.value, dtype=np.float64)
            label = class_labels[int(np.argmax(counts))] if counts.size else "?"
            lines.append((depth, f"{edge}Leaf {label} · {int(node.n_samples)} rows"))
            return
        name = feature_names[int(node.feature)]
        lines.append((depth, f"{edge}{display_name(name)} ≤ {float(node.threshold):.4g}"))
        walk(node.left, depth + 1, "Yes · ")
        walk(node.right, depth + 1, "No · ")

    walk(root, 0, "")
    return lines


def input_step(feature: str) -> float:
    """Widget step for one raw feature."""
    return float(STEPS.get(feature, 0.1))
