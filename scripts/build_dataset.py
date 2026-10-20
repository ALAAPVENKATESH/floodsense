"""Build data/processed/dataset.csv from data/raw, or write a synthetic table.

Measured files stay measured. If no CSV is in the raw folder, this script
writes flood_synthetic.csv and dataset.csv and says so on stderr.
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
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import config
import generate_synthetic_data as synthetic
from flood_rf.preprocessing import (
    bin_unit_score,
    blank_physical_extremes,
    drop_duplicate_rows,
    normalize_label,
    remove_invalid_rows,
)


def normalize_columns(frame: pd.DataFrame) -> pd.DataFrame:
    renamed = []
    for column in frame.columns:
        key = str(column).strip().lower().replace(" ", "_").replace("-", "_")
        renamed.append(config.COLUMN_ALIASES.get(key, key))
    if len(set(renamed)) != len(renamed):
        raise ValueError(f"Duplicate column names after normalization: {renamed}")
    out = frame.copy()
    out.columns = renamed
    return out


def ensure_location_id(frame: pd.DataFrame) -> pd.DataFrame:
    if "location_id" in frame.columns:
        return frame
    out = frame.copy()
    if {"district", "period"}.issubset(out.columns):
        out["location_id"] = out["district"].astype(str) + "|" + out["period"].astype(str)
    elif {"latitude", "longitude", "period"}.issubset(out.columns):
        out["location_id"] = (
            out["latitude"].round(4).astype(str)
            + "|"
            + out["longitude"].round(4).astype(str)
            + "|"
            + out["period"].astype(str)
        )
    return out


def merge_frames(frames: list[pd.DataFrame]) -> tuple[pd.DataFrame, str, list[str]]:
    frames = [ensure_location_id(frame) for frame in frames]
    if len(frames) == 1:
        return frames[0], "single file", []
    for key in config.MERGE_KEYS:
        if all(key in frame.columns for frame in frames):
            merged = frames[0]
            dropped: list[str] = []
            for other in frames[1:]:
                overlap = [column for column in other.columns if column != key and column in merged.columns]
                dropped.extend(overlap)
                merged = merged.merge(other.drop(columns=overlap), on=key, how="inner")
            return merged, f"inner join on {key}", dropped
    if len({len(frame) for frame in frames}) == 1:
        merged = pd.concat(frames, axis=1)
        merged = merged.loc[:, ~merged.columns.duplicated()]
        return merged, "column-wise concat (same row count, no shared key)", []
    raise SystemExit(
        "Cannot merge raw CSVs: no shared location_id or district, and the files have different lengths."
    )


def _coerce(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    numeric = config.BASE_FEATURES + config.OPTIONAL_FEATURES + config.LABEL_SCORE_COLUMNS
    for column in numeric:
        if column in out.columns:
            out[column] = pd.to_numeric(out[column], errors="coerce")
    return out


def apply_labels(frame: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    info = {"excluded_features": [], "label_rule": "", "label_score_column": None}
    out = frame.copy()
    if config.TARGET in out.columns and out[config.TARGET].notna().any():
        out[config.TARGET] = out[config.TARGET].map(normalize_label)
        info["label_rule"] = "risk_class column supplied by the source file"
        return out, info
    for column in ("flood_hazard_score", "historical_flood"):
        if column not in out.columns:
            continue
        values = pd.to_numeric(out[column], errors="coerce")
        finite = values.dropna()
        if finite.empty:
            continue
        if float(finite.min()) >= 0 and float(finite.max()) <= 1:
            out[config.TARGET] = bin_unit_score(values)
            out = out.drop(columns=[column])
            info["excluded_features"].append(column)
            info["label_rule"] = (
                f"Fixed cuts on {column}: <= {config.UNIT_SCORE_THRESHOLDS[0]} LOW, "
                f"<= {config.UNIT_SCORE_THRESHOLDS[1]} MODERATE, else HIGH. "
                f"{column} was removed from the predictors."
            )
            return out, info
        info["label_score_column"] = column
        info["label_rule"] = (
            f"{column} is not on a 0-1 scale. scripts/train.py bins it with quantiles "
            f"{config.TRAIN_SCORE_QUANTILES} fit on the training split only, then drops it."
        )
        return out, info
    raise SystemExit("No risk_class, flood_hazard_score, or historical_flood column to label rows.")


def merge_external(frame: pd.DataFrame, external: pd.DataFrame) -> pd.DataFrame:
    """Left-join external features on row_id. Row count and order stay put."""
    if "row_id" not in frame.columns or external is None or external.empty:
        return frame
    order = frame["row_id"].tolist()
    extra = external.drop_duplicates("row_id", keep="first")
    columns = ["row_id"] + [column for column in extra.columns if column not in frame.columns]
    merged = frame.merge(extra[columns], on="row_id", how="left")
    if len(merged) != len(frame) or merged["row_id"].tolist() != order:
        raise RuntimeError("external feature merge changed the row count or order")
    return merged


def _keep(frame: pd.DataFrame) -> pd.DataFrame:
    wanted = (
        config.ID_COLUMNS
        + config.BASE_FEATURES
        + config.OPTIONAL_FEATURES
        + config.LABEL_SCORE_COLUMNS
        + [config.TARGET, config.SOURCE_COLUMN]
    )
    columns = [column for column in wanted if column in frame.columns]
    return frame[columns].copy()


def _write_meta(path: Path, meta: dict) -> None:
    path.write_text(json.dumps(meta, indent=2), encoding="utf-8")


def build(raw_dir: Path | None = None, out_dir: Path | None = None, n_synthetic: int = synthetic.N_DEFAULT, seed: int | None = None) -> dict:
    raw_dir = config.DATA_RAW if raw_dir is None else Path(raw_dir)
    out_dir = config.DATA_PROCESSED if out_dir is None else Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    raw_dir.mkdir(parents=True, exist_ok=True)
    csvs = sorted(path for path in raw_dir.glob("*.csv") if path.is_file())
    dataset_path = out_dir / "dataset.csv"
    meta_path = out_dir / "dataset_meta.json"

    if not csvs:
        print(config.SYNTHETIC_WARNING, file=sys.stderr)
        frame = synthetic.generate(n_synthetic, config.RANDOM_STATE if seed is None else seed)
        synthetic_path = out_dir / "flood_synthetic.csv"
        frame.to_csv(synthetic_path, index=False)
        frame.to_csv(dataset_path, index=False)
        counts = frame["risk_class"].value_counts().to_dict()
        meta = {
            "data_source": "SYNTHETIC",
            "warning": config.SYNTHETIC_WARNING,
            "n_rows": int(len(frame)),
            "class_counts": {str(k): int(v) for k, v in counts.items()},
            "label_rule": synthetic.LABEL_RULE,
            "files": [],
            "generator_seed": config.RANDOM_STATE if seed is None else seed,
            "excluded_features": [],
            "dropped_invalid": 0,
            "dropped_duplicates": 0,
        }
        _write_meta(meta_path, meta)
        return meta

    frames = []
    for path in csvs:
        frame = pd.read_csv(path)
        if frame.empty:
            raise SystemExit(f"{path.name} has no rows")
        frames.append(normalize_columns(frame))
    merged, how, dropped_overlap = merge_frames(frames)
    merged = _coerce(merged)
    merged, label_info = apply_labels(merged)
    tags = set()
    if config.SOURCE_COLUMN in merged.columns:
        tags = set(merged[config.SOURCE_COLUMN].astype(str).str.upper())
    if tags == {"SYNTHETIC"}:
        source = "SYNTHETIC"
        print(config.SYNTHETIC_WARNING, file=sys.stderr)
    elif tags == {config.BENCHMARK_SOURCE}:
        source = config.BENCHMARK_SOURCE
        merged[config.SOURCE_COLUMN] = config.BENCHMARK_SOURCE
        label_info["label_rule"] = config.BENCHMARK_LABEL_RULE
        print(config.BENCHMARK_WARNING, file=sys.stderr)
    elif tags == {config.CLASSIFICATION_SOURCE}:
        source = config.CLASSIFICATION_SOURCE
        merged[config.SOURCE_COLUMN] = config.CLASSIFICATION_SOURCE
        label_info["label_rule"] = config.CLASSIFICATION_LABEL_RULE
        print(config.CLASSIFICATION_WARNING, file=sys.stderr)
    else:
        merged[config.SOURCE_COLUMN] = "REAL"
        source = "REAL"
    external_path = out_dir / "external_features.csv"
    if external_path.exists() and "row_id" in merged.columns:
        merged = merge_external(merged, pd.read_csv(external_path))
        print(f"merged {external_path.name} on row_id", flush=True)
    elif "row_id" in merged.columns:
        print("no data/processed/external_features.csv; external rainfall and population stay absent", flush=True)
    if "year" not in merged.columns and "period" in merged.columns:
        years = pd.to_numeric(merged["period"], errors="coerce")
        if years.notna().any() and years.dropna().between(1800, 2100).all():
            merged["year"] = years
    ranges = dict(config.FEATURE_RANGES)
    if label_info.get("label_score_column"):
        ranges.pop(label_info["label_score_column"], None)
    merged, n_masked = blank_physical_extremes(merged)
    merged, n_invalid = remove_invalid_rows(merged, ranges)
    if config.TARGET in merged.columns:
        merged = merged[merged[config.TARGET].notna()].reset_index(drop=True)
        unknown = ~merged[config.TARGET].isin(config.CLASS_LABELS)
        n_unknown = int(unknown.sum())
        merged = merged.loc[~unknown].reset_index(drop=True)
    else:
        n_unknown = 0
    duplicate_on = [column for column in merged.columns if column not in {"row_id", "pop_source_year"}]
    merged, n_dupes = drop_duplicate_rows(merged, subset=duplicate_on or None)
    merged = _keep(merged)
    if merged.empty:
        raise SystemExit("No rows left after cleaning the raw CSVs.")
    merged.to_csv(dataset_path, index=False)
    synthetic_path = out_dir / "flood_synthetic.csv"
    if source != "SYNTHETIC" and synthetic_path.exists():
        synthetic_path.unlink()
    counts = (
        merged[config.TARGET].value_counts().to_dict()
        if config.TARGET in merged.columns
        else {}
    )
    meta = {
        "data_source": source,
        "warning": config.warning_for(source),
        "coverage": (
            config.BENCHMARK_COVERAGE
            if source == config.BENCHMARK_SOURCE
            else config.CLASSIFICATION_COVERAGE
            if source == config.CLASSIFICATION_SOURCE
            else ""
        ),
        "n_rows": int(len(merged)),
        "class_counts": {str(k): int(v) for k, v in counts.items()},
        "label_rule": label_info["label_rule"],
        "label_score_column": label_info.get("label_score_column"),
        "excluded_features": label_info["excluded_features"],
        "files": [path.name for path in csvs],
        "merge": how,
        "dropped_overlapping_columns": dropped_overlap,
        "dropped_invalid": n_invalid + n_unknown,
        "masked_extremes": n_masked,
        "dropped_duplicates": n_dupes,
    }
    note_path = raw_dir / "urbanfloodbench_note.json"
    if source == config.BENCHMARK_SOURCE and note_path.exists():
        note = json.loads(note_path.read_text(encoding="utf-8"))
        for key in ("rows_before_sample", "sample_cap", "models", "events", "units"):
            if key in note:
                meta[key] = note[key]
    _write_meta(meta_path, meta)
    return meta


def main() -> None:
    meta = build()
    print(f"data_source={meta['data_source']} rows={meta['n_rows']}")
    print(meta["label_rule"])
    if meta["data_source"] != "REAL":
        print(config.warning_for(meta["data_source"]), file=sys.stderr)


if __name__ == "__main__":
    main()
