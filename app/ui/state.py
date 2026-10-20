"""Cached reads for the app. The training code is not called from here."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import pandas as pd
import streamlit as st

import config
from flood_rf.persistence import load_model
from flood_rf.preprocessing import split_positions, stratified_train_test_split


def mode() -> str:
    """Active appearance. The radio stores Light or Dark."""
    return "dark" if st.session_state.get("theme_name") == "Dark" else "light"


@st.cache_data(show_spinner=False)
def _read_json(path: str, mtime: float) -> dict:
    del mtime
    return json.loads(Path(path).read_text(encoding="utf-8"))


@st.cache_resource(show_spinner=False)
def _load_model(path: str, mtime: float):
    del mtime
    return load_model(Path(path))


@st.cache_data(show_spinner=False)
def _read_csv(path: str, mtime: float) -> pd.DataFrame:
    del mtime
    return pd.read_csv(path)


@st.cache_data(show_spinner=False)
def _training_frame(path: str, mtime: float, seed: int, test_size: float) -> pd.DataFrame:
    """Rows from the same seeded split the trainer uses. Used only for percentiles."""
    del mtime
    frame = pd.read_csv(path)
    if {"latitude", "longitude"} <= set(frame.columns):
        train_idx, _test_idx = split_positions(frame, "spatial", seed, test_size)
        return frame.iloc[train_idx].reset_index(drop=True)
    if config.TARGET not in frame.columns or not frame[config.TARGET].notna().any():
        train_idx, _test_idx = split_positions(frame, "random", seed, test_size)
        return frame.iloc[train_idx].reset_index(drop=True)
    target = frame[config.TARGET]
    predictors = frame.drop(columns=[config.TARGET])
    train, _, _, _ = stratified_train_test_split(
        predictors, target, test_size=test_size, random_state=seed
    )
    return train


def metrics() -> dict | None:
    """Saved holdout report, or None when train.py has not been run."""
    path = config.METRICS_PATH
    if not path.exists():
        return None
    return _read_json(str(path), path.stat().st_mtime)


def importance() -> dict | None:
    """Saved importance lists, or None."""
    path = config.IMPORTANCE_PATH
    if not path.exists():
        return None
    return _read_json(str(path), path.stat().st_mtime)


def dataset_meta() -> dict:
    """Build-script notes. Empty when the file is absent."""
    path = config.DATASET_META
    if not path.exists():
        return {}
    return _read_json(str(path), path.stat().st_mtime)


def model():
    """Fitted forest wrapper, or None when models/model.json is missing."""
    path = config.MODEL_PATH
    if not path.exists():
        return None
    return _load_model(str(path), path.stat().st_mtime)


def table() -> pd.DataFrame | None:
    """Processed modeling table, or None."""
    path = config.DATASET_CSV
    if not path.exists():
        return None
    return _read_csv(str(path), path.stat().st_mtime)


def training_rows() -> pd.DataFrame | None:
    """Training split of the processed table, for percentile markers."""
    path = config.DATASET_CSV
    if not path.exists():
        return None
    return _training_frame(str(path), path.stat().st_mtime, config.RANDOM_STATE, config.TEST_SIZE)


def source_name(report: dict | None) -> str:
    """REAL, SYNTHETIC, or UNKNOWN."""
    if not report:
        return "UNKNOWN"
    return str(report.get("data_source") or "UNKNOWN")


def tree_count(report: dict | None) -> int | None:
    """Estimator count recorded at train time."""
    if not report:
        return None
    return report.get("hyperparameters", {}).get("n_estimators")


def trained_at() -> str:
    """Local timestamp of the saved model file."""
    path = config.MODEL_PATH
    if not path.exists():
        return "not trained"
    stamp = datetime.fromtimestamp(path.stat().st_mtime)
    return stamp.strftime("%d %b %Y, %H:%M")
