"""Build data/raw/urbanfloodbench.csv from the Kaggle UrbanFloodBench archive.

The archive is HEC-RAS output (competition models 1 and 2, US customary units).
One row is one surface cell in one rainfall event. Water level is used only
as the label. It is not a predictor.

Peak depth (m) = max(0, max water surface - cell elevation) * 0.3048.
LOW < 0.15 m, MODERATE < 0.50 m, else HIGH.
The from-scratch forest cannot train on every cell, so the table is
stratified down to config.BENCHMARK_MAX_ROWS with seed 42.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import config

FT_TO_M = 0.3048
IN_TO_MM = 25.4
DATASET = "jiayulim/urbanfloodbench"


def depth_to_class(depth_m: pd.Series) -> pd.Series:
    low, high = config.BENCHMARK_DEPTH_M
    labels = np.where(depth_m < low, "LOW", np.where(depth_m < high, "MODERATE", "HIGH"))
    return pd.Series(labels, index=depth_m.index)


def stratified_cap(frame: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    if len(frame) <= n:
        return frame.reset_index(drop=True)
    groups = [(label, group) for label, group in frame.groupby("risk_class", sort=False)]
    sizes = [max(1, int(round(n * len(group) / len(frame)))) for _, group in groups]
    while sum(sizes) > n:
        index = max(range(len(sizes)), key=lambda i: sizes[i])
        if sizes[index] == 1:
            break
        sizes[index] -= 1
    while sum(sizes) < n:
        index = max(range(len(sizes)), key=lambda i: len(groups[i][1]) - sizes[i])
        if sizes[index] >= len(groups[index][1]):
            break
        sizes[index] += 1
    rng = np.random.RandomState(seed)
    parts = []
    for (label, group), size in zip(groups, sizes):
        take = min(size, len(group))
        chosen = rng.choice(len(group), size=take, replace=False)
        parts.append(group.iloc[chosen])
    out = pd.concat(parts, ignore_index=True)
    order = rng.permutation(len(out))
    return out.iloc[order].reset_index(drop=True)


def _columns(path: Path) -> list[str]:
    return list(pd.read_csv(path, nrows=0).columns)


def _pick(columns: list[str], names: list[str]) -> str | None:
    lookup = {column.lower(): column for column in columns}
    for name in names:
        if name in lookup:
            return lookup[name]
    return None


def event_hours(path: Path) -> float:
    frame = pd.read_csv(path)
    n = len(frame)
    if n < 2:
        return 5 / 60
    named = [column for column in frame.columns if column.lower() in {"timestamp", "datetime", "date_time"}]
    time_col = named[0] if named else next(
        (
            column
            for column in frame.columns
            if "idx" not in column.lower()
            and "step" not in column.lower()
            and any(token in column.lower() for token in ("time", "date"))
        ),
        None,
    )
    if time_col is not None:
        parsed = pd.to_datetime(frame[time_col], errors="coerce").dropna().sort_values()
        if len(parsed) >= 2:
            steps = parsed.diff().dt.total_seconds().dropna()
            step_h = float(steps.median()) / 3600 if len(steps) else 5 / 60
            span_h = (parsed.iloc[-1] - parsed.iloc[0]).total_seconds() / 3600
            return max(span_h + step_h, step_h)
    return n * 5 / 60


def cell_slope(folder: Path) -> pd.Series | None:
    index_path = folder / "2d_edge_index.csv"
    slope_path = folder / "2d_edges_static.csv"
    if not index_path.exists() or not slope_path.exists():
        return None
    index = pd.read_csv(index_path)
    slopes = pd.read_csv(slope_path)
    from_col = _pick(list(index.columns), ["from_node", "from_node_idx", "source"])
    to_col = _pick(list(index.columns), ["to_node", "to_node_idx", "target"])
    slope_col = _pick(list(slopes.columns), ["slope"])
    if not from_col or not to_col or not slope_col:
        return None
    edge_index = _pick(list(index.columns), ["edge_idx"])
    edge_slope = _pick(list(slopes.columns), ["edge_idx"])
    if edge_index and edge_slope:
        values = index[[edge_index]].merge(
            slopes[[edge_slope, slope_col]],
            left_on=edge_index,
            right_on=edge_slope,
            how="left",
        )[slope_col].to_numpy(dtype=np.float64)
    elif len(index) == len(slopes):
        values = slopes[slope_col].to_numpy(dtype=np.float64)
    else:
        return None
    ends = pd.DataFrame(
        {
            "from_node": index[from_col].to_numpy(),
            "to_node": index[to_col].to_numpy(),
            "slope": np.abs(values),
        }
    )
    long = pd.concat(
        [
            ends.rename(columns={"from_node": "node_idx"})[["node_idx", "slope"]],
            ends.rename(columns={"to_node": "node_idx"})[["node_idx", "slope"]],
        ],
        ignore_index=True,
    )
    median = long.groupby("node_idx")["slope"].median()
    # US edge files store rise/run. A median above 1.5 is already degrees.
    if float(median.median()) > 1.5:
        return median.clip(0, 90)
    return np.degrees(np.arctan(median))


def aggregate_dynamic(path: Path) -> pd.DataFrame | None:
    columns = _columns(path)
    node_col = _pick(columns, ["node_idx", "node_id"])
    rain_col = _pick(columns, ["rainfall"])
    level_col = _pick(columns, ["water_level"])
    if not node_col or not rain_col or not level_col:
        return None
    rain = None
    level = None
    for chunk in pd.read_csv(path, usecols=[node_col, rain_col, level_col], chunksize=1_000_000):
        chunk = chunk.rename(columns={node_col: "node_idx", rain_col: "rainfall", level_col: "water_level"})
        grouped = chunk.groupby("node_idx", sort=False)
        rain_sum = grouped["rainfall"].sum()
        level_max = grouped["water_level"].max()
        rain = rain_sum if rain is None else rain.add(rain_sum, fill_value=0.0)
        if level is None:
            level = level_max
        else:
            level = pd.concat([level.rename("a"), level_max.rename("b")], axis=1).max(axis=1)
    out = pd.DataFrame({"rainfall_in": rain, "water_level_ft": level})
    out.index.name = "node_idx"
    return out.reset_index()


def load_static(folder: Path) -> pd.DataFrame:
    frame = pd.read_csv(folder / "2d_nodes_static.csv")
    node_col = _pick(list(frame.columns), ["node_idx", "node_id"])
    elev_col = _pick(list(frame.columns), ["elevation", "centroid_elevation"])
    if not node_col or not elev_col:
        raise SystemExit(f"{folder} static file has no node index or elevation")
    out = pd.DataFrame(
        {
            "node_idx": frame[node_col].to_numpy(),
            "elevation_ft": pd.to_numeric(frame[elev_col], errors="coerce").to_numpy(),
        }
    )
    slope = cell_slope(folder)
    if slope is not None:
        out = out.merge(slope.rename("slope").reset_index(), on="node_idx", how="left")
    return out


def iter_splits(root: Path):
    for static in sorted(root.rglob("2d_nodes_static.csv")):
        folder = static.parent
        if folder.name.lower() != "train":
            continue
        events = sorted(path for path in folder.iterdir() if path.is_dir() and path.name.lower().startswith("event"))
        if events:
            yield folder, events


def build_table(root: Path) -> tuple[pd.DataFrame, dict]:
    rows = []
    models = []
    events = []
    for folder, event_dirs in iter_splits(root):
        model_name = folder.parent.name
        models.append(model_name)
        static = load_static(folder)
        print(f"{model_name}: {len(static)} cells, {len(event_dirs)} train events", flush=True)
        for event_dir in event_dirs:
            dynamic_path = event_dir / "2d_nodes_dynamic_all.csv"
            if not dynamic_path.exists():
                continue
            dynamic = aggregate_dynamic(dynamic_path)
            if dynamic is None:
                print(f"skip {dynamic_path} (no water_level)", flush=True)
                continue
            hours = event_hours(event_dir / "timesteps.csv") if (event_dir / "timesteps.csv").exists() else 1.0
            merged = static.merge(dynamic, on="node_idx", how="inner")
            depth_m = (merged["water_level_ft"] - merged["elevation_ft"]).clip(lower=0) * FT_TO_M
            rainfall_mm = merged["rainfall_in"] * IN_TO_MM
            piece = pd.DataFrame(
                {
                    "location_id": model_name + "|" + event_dir.name + "|" + merged["node_idx"].astype(str),
                    "district": model_name,
                    "period": event_dir.name,
                    "rainfall": rainfall_mm,
                    "rainfall_intensity": rainfall_mm / hours,
                    "elevation": merged["elevation_ft"] * FT_TO_M,
                    "risk_class": depth_to_class(depth_m),
                    "data_source": config.BENCHMARK_SOURCE,
                }
            )
            if "slope" in merged.columns:
                piece["slope"] = merged["slope"].to_numpy()
            rows.append(piece)
            events.append(f"{model_name}/{event_dir.name}")
            print(f"  {event_dir.name}: {len(piece)} cells, {hours:.2f} h", flush=True)
    if not rows:
        raise SystemExit(f"No train events with water_level under {root}")
    table = pd.concat(rows, ignore_index=True)
    note = {
        "rows_before_sample": int(len(table)),
        "sample_cap": int(config.BENCHMARK_MAX_ROWS),
        "models": models,
        "events": events,
        "units": "US customary converted: elevation ft to m, rainfall inches to mm, edge slope rise/run to degrees",
    }
    table = stratified_cap(table, config.BENCHMARK_MAX_ROWS, config.RANDOM_STATE)
    return table, note


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=Path, default=None, help="Extracted archive. Downloads with kagglehub if omitted.")
    args = parser.parse_args()
    if args.path is None:
        import kagglehub

        root = Path(kagglehub.dataset_download(DATASET))
    else:
        root = args.path
    table, note = build_table(root)
    config.DATA_RAW.mkdir(parents=True, exist_ok=True)
    csv_path = config.DATA_RAW / "urbanfloodbench.csv"
    note_path = config.DATA_RAW / "urbanfloodbench_note.json"
    table.to_csv(csv_path, index=False)
    note_path.write_text(json.dumps(note, indent=2), encoding="utf-8")
    print(f"wrote {csv_path} rows={len(table)} from {note['rows_before_sample']}")
    print(table["risk_class"].value_counts().to_string())
    print(config.BENCHMARK_WARNING, file=sys.stderr)


if __name__ == "__main__":
    main()
