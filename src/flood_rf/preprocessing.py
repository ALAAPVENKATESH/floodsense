"""Imputation, range checks, encoding, and seeded splits.

Every statistic used at transform time is stored on `fit`. Fit the preprocessor
on training rows only, then transform validation and test rows with that object.
"""

from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd

import config


def drop_duplicate_rows(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    before = len(df)
    cleaned = df.drop_duplicates().reset_index(drop=True)
    return cleaned, before - len(cleaned)


def invalid_mask(df: pd.DataFrame, ranges: dict | None = None) -> pd.Series:
    """True where a known numeric column falls outside a physical bound."""
    ranges = config.FEATURE_RANGES if ranges is None else ranges
    mask = pd.Series(False, index=df.index)
    for col, (lo, hi) in ranges.items():
        if col not in df.columns:
            continue
        values = pd.to_numeric(df[col], errors="coerce")
        present = df[col].notna()
        mask = mask | (present & ((values < lo) | (values > hi) | values.isna()))
    return mask


def remove_invalid_rows(df: pd.DataFrame, ranges: dict | None = None) -> tuple[pd.DataFrame, int]:
    mask = invalid_mask(df, ranges)
    return df.loc[~mask].reset_index(drop=True), int(mask.sum())


def normalize_label(value) -> str | None:
    if pd.isna(value):
        return None
    key = str(value).strip().upper()
    return config.LABEL_SYNONYMS.get(key)


def bin_unit_score(score: pd.Series) -> pd.Series:
    """Map a 0-1 score with fixed cuts. The cuts do not depend on the sample."""
    low, mid = config.UNIT_SCORE_THRESHOLDS
    values = pd.to_numeric(score, errors="coerce")
    labels = pd.Series(np.where(values <= mid, "MODERATE", "HIGH"), index=score.index)
    labels = labels.mask(values <= low, "LOW")
    labels = labels.mask(values.isna(), np.nan)
    return labels


def bin_score_train_quantiles(train_score, other_score, quantiles: tuple[float, float] | None = None):
    """Bin a score using quantiles fit on `train_score` only."""
    quantiles = config.TRAIN_SCORE_QUANTILES if quantiles is None else quantiles
    train_values = np.asarray(pd.to_numeric(pd.Series(train_score), errors="coerce"), dtype=np.float64)
    train_values = train_values[np.isfinite(train_values)]
    if len(train_values) == 0:
        raise ValueError("Cannot bin a score with no finite training values")
    q1, q2 = np.quantile(train_values, list(quantiles))

    def apply(score) -> np.ndarray:
        values = np.asarray(pd.to_numeric(pd.Series(score), errors="coerce"), dtype=np.float64)
        labels = np.full(len(values), "HIGH", dtype=object)
        labels[values <= q2] = "MODERATE"
        labels[values <= q1] = "LOW"
        labels[~np.isfinite(values)] = None
        return labels

    return apply(train_score), apply(other_score), (float(q1), float(q2))


def stratified_indices(y, test_size: float = 0.2, random_state: int = 42) -> tuple[np.ndarray, np.ndarray]:
    """Seeded stratified holdout. Each class keeps at least one training row when possible."""
    if not 0 < test_size < 1:
        raise ValueError("test_size must be between 0 and 1")
    y = np.asarray(y)
    rng = np.random.RandomState(random_state)
    train_parts: list[np.ndarray] = []
    test_parts: list[np.ndarray] = []
    for cls in np.unique(y):
        idx = np.flatnonzero(y == cls)
        rng.shuffle(idx)
        if len(idx) < 2:
            train_parts.append(idx)
            continue
        n_test = int(round(len(idx) * test_size))
        n_test = min(max(n_test, 1), len(idx) - 1)
        test_parts.append(idx[:n_test])
        train_parts.append(idx[n_test:])
    train_idx = np.concatenate(train_parts)
    test_idx = np.concatenate(test_parts) if test_parts else np.array([], dtype=int)
    rng.shuffle(train_idx)
    rng.shuffle(test_idx)
    return train_idx, test_idx


def _take(obj, idx: np.ndarray):
    if isinstance(obj, (pd.DataFrame, pd.Series)):
        return obj.iloc[idx].reset_index(drop=True)
    return np.asarray(obj)[idx]


def stratified_train_test_split(X, y, test_size: float = 0.2, random_state: int = 42):
    train_idx, test_idx = stratified_indices(y, test_size=test_size, random_state=random_state)
    return _take(X, train_idx), _take(X, test_idx), _take(y, train_idx), _take(y, test_idx)


def stratified_kfold_indices(y, n_splits: int = 5, random_state: int = 42):
    """Yield (train_idx, val_idx) pairs. A class smaller than `n_splits` misses some folds."""
    y = np.asarray(y)
    if n_splits < 2:
        raise ValueError("n_splits must be at least 2")
    if n_splits > len(y):
        raise ValueError("n_splits cannot exceed the number of rows")
    rng = np.random.RandomState(random_state)
    folds: list[list[int]] = [[] for _ in range(n_splits)]
    for cls in np.unique(y):
        idx = np.flatnonzero(y == cls)
        rng.shuffle(idx)
        for i, sample_i in enumerate(idx):
            folds[i % n_splits].append(int(sample_i))
    folded = [np.array(sorted(fold), dtype=int) for fold in folds]
    for i in range(n_splits):
        val = folded[i]
        others = [folded[j] for j in range(n_splits) if j != i and len(folded[j])]
        if len(val) == 0 or not others:
            continue
        yield np.concatenate(others), val


class Preprocessor:
    """Median / mode imputation, optional IQR clipping, and label-encoding of categories.

    Outlier clipping is off by default: extreme rainfall is part of the flood signal.
    Unordered categories are integer codes in sorted order. Unknown test categories
    become -1. That encoding is only a fallback for an extra text column in a real CSV.
    """

    def __init__(self, clip_outliers: bool = False):
        self.clip_outliers = clip_outliers
        self.medians_: dict[str, float] = {}
        self.modes_: dict[str, object] = {}
        self.categories_: dict[str, dict] = {}
        self.clip_bounds_: dict[str, tuple[float, float]] = {}
        self.columns_: list[str] = []
        self.numeric_columns_: list[str] = []
        self.categorical_columns_: list[str] = []

    def fit(self, X: pd.DataFrame) -> "Preprocessor":
        frame = X.copy()
        self.columns_ = list(frame.columns)
        self.numeric_columns_ = []
        self.categorical_columns_ = []
        self.medians_ = {}
        self.modes_ = {}
        self.categories_ = {}
        self.clip_bounds_ = {}
        for col in self.columns_:
            if pd.api.types.is_numeric_dtype(frame[col]):
                self.numeric_columns_.append(col)
                median = frame[col].median(skipna=True)
                self.medians_[col] = 0.0 if pd.isna(median) else float(median)
                if self.clip_outliers:
                    q1 = float(frame[col].quantile(0.25))
                    q3 = float(frame[col].quantile(0.75))
                    iqr = q3 - q1
                    self.clip_bounds_[col] = (q1 - 1.5 * iqr, q3 + 1.5 * iqr)
            else:
                self.categorical_columns_.append(col)
                observed = frame[col].dropna()
                if observed.empty:
                    self.modes_[col] = "MISSING"
                else:
                    self.modes_[col] = observed.mode(dropna=True).iloc[0]
                cats = sorted({str(v) for v in observed.tolist()})
                self.categories_[col] = {cat: i for i, cat in enumerate(cats)}
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if not self.columns_:
            raise RuntimeError("Preprocessor.fit must be called before transform")
        frame = X.copy()
        for col in self.columns_:
            if col not in frame.columns:
                frame[col] = np.nan
        frame = frame[self.columns_].copy()
        for col in self.numeric_columns_:
            values = pd.to_numeric(frame[col], errors="coerce")
            values = values.fillna(self.medians_[col])
            if col in self.clip_bounds_:
                lo, hi = self.clip_bounds_[col]
                values = values.clip(lo, hi)
            frame[col] = values.astype(float)
        for col in self.categorical_columns_:
            filled = frame[col].where(frame[col].notna(), self.modes_[col])
            mapping = self.categories_[col]
            frame[col] = [float(mapping.get(str(v), -1)) for v in filled]
        return frame

    def fit_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return self.fit(X).transform(X)

    def to_dict(self) -> dict:
        return {
            "clip_outliers": self.clip_outliers,
            "medians": self.medians_,
            "modes": {k: (None if pd.isna(v) else v) for k, v in self.modes_.items()},
            "categories": self.categories_,
            "clip_bounds": {k: list(v) for k, v in self.clip_bounds_.items()},
            "columns": self.columns_,
            "numeric_columns": self.numeric_columns_,
            "categorical_columns": self.categorical_columns_,
        }

    @classmethod
    def from_dict(cls, payload: dict) -> "Preprocessor":
        pre = cls(clip_outliers=bool(payload.get("clip_outliers", False)))
        pre.medians_ = {k: float(v) for k, v in payload["medians"].items()}
        pre.modes_ = dict(payload["modes"])
        pre.categories_ = {k: {str(a): int(b) for a, b in v.items()} for k, v in payload["categories"].items()}
        pre.clip_bounds_ = {k: (float(v[0]), float(v[1])) for k, v in payload["clip_bounds"].items()}
        pre.columns_ = list(payload["columns"])
        pre.numeric_columns_ = list(payload["numeric_columns"])
        pre.categorical_columns_ = list(payload["categorical_columns"])
        return pre


def encode_labels(y, labels: Iterable[str] | None = None) -> tuple[np.ndarray, list[str]]:
    labels = list(config.CLASS_LABELS if labels is None else labels)
    lookup = {label: i for i, label in enumerate(labels)}
    encoded = []
    for raw in y:
        if isinstance(raw, (int, np.integer)) and not isinstance(raw, bool) and int(raw) in lookup.values():
            encoded.append(int(raw))
            continue
        key = str(raw).strip().upper()
        if key in lookup:
            encoded.append(lookup[key])
        elif key in config.LABEL_SYNONYMS and config.LABEL_SYNONYMS[key] in lookup:
            encoded.append(lookup[config.LABEL_SYNONYMS[key]])
        else:
            raise ValueError(f"Unknown class label: {raw!r}")
    return np.asarray(encoded, dtype=np.int64), labels
