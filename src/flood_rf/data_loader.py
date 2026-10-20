"""Load the processed table and report whether it is measured or simulated."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

import config


def load_processed(path: Path | None = None) -> pd.DataFrame:
    path = config.DATASET_CSV if path is None else Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {path}. Run: python scripts/build_dataset.py"
        )
    return pd.read_csv(path)


def source_flag(df: pd.DataFrame) -> str:
    if config.SOURCE_COLUMN not in df.columns:
        return "UNKNOWN"
    values = set(df[config.SOURCE_COLUMN].astype(str).str.upper().unique())
    if values == {"SYNTHETIC"}:
        return "SYNTHETIC"
    if "SYNTHETIC" in values:
        return "MIXED"
    return "REAL"


def model_feature_columns(df: pd.DataFrame, drop: list[str] | None = None) -> list[str]:
    """Columns the forest may see. Absent optional columns are omitted, never filled in here."""
    blocked = set(drop or [])
    blocked.update([config.TARGET, config.SOURCE_COLUMN, *config.ID_COLUMNS, *config.LABEL_SCORE_COLUMNS])
    chosen = []
    for col in config.BASE_FEATURES + config.OPTIONAL_FEATURES:
        if col in df.columns and col not in blocked:
            chosen.append(col)
    return chosen
