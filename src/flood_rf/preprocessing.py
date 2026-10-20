"""Imputation, range checks, encoding, and seeded splits.

Every statistic used at transform time is stored on `fit`. Fit the preprocessor
on training rows only, then transform validation and test rows with that object.
"""

from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd

import config


def drop_duplicate_rows(df: pd.DataFrame, subset: list[str] | None = None) -> tuple[pd.DataFrame, int]:
    before = len(df)
    cleaned = df.drop_duplicates(subset=subset).reset_index(drop=True)
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


def grid_cell_ids(lat, lon, cell_deg: float = 5.0) -> np.ndarray:
    """Stable block id ``lat_bin:lon_bin`` from floor(coord / cell_deg)."""
    if cell_deg <= 0:
        raise ValueError("cell_deg must be positive")
    lat_v = np.asarray(pd.to_numeric(pd.Series(lat), errors="coerce"), dtype=np.float64)
    lon_v = np.asarray(pd.to_numeric(pd.Series(lon), errors="coerce"), dtype=np.float64)
    lat_bin = np.floor(lat_v / cell_deg)
    lon_bin = np.floor(lon_v / cell_deg)
    ids = np.empty(len(lat_v), dtype=object)
    for i, (lat_i, lon_i) in enumerate(zip(lat_bin, lon_bin)):
        if not np.isfinite(lat_i) or not np.isfinite(lon_i):
            ids[i] = "missing"
        else:
            ids[i] = f"{int(lat_i)}:{int(lon_i)}"
    return ids


def _group_index(groups: np.ndarray) -> tuple[np.ndarray, dict]:
    groups = np.asarray(groups)
    uniq = np.unique(groups)
    return uniq, {group: np.flatnonzero(groups == group) for group in uniq}


def group_train_test_split(groups, test_size, seed, y=None):
    """Hold out whole groups. Class balance is best effort when `y` is given.

    Tolerance: the splitter tries to land within 25% of the target test-row
    count, and, when labels are supplied, prefers groups that move each
    class toward its target test count. A group is never split. A class share
    can still miss the population share when one cell is large; that gap is
    reported by the caller, not forced.
    """
    groups = np.asarray(groups)
    n = len(groups)
    if n < 2:
        raise ValueError("group split needs at least two rows")
    uniq, index = _group_index(groups)
    rng = np.random.RandomState(int(seed))
    order = uniq.copy()
    rng.shuffle(order)
    target = int(round(n * float(test_size)))
    target = min(max(target, 1), n - 1)
    y_arr = None if y is None else np.asarray(list(y))
    classes = [] if y_arr is None else list(np.unique(y_arr))
    target_c = {cls: float(np.sum(y_arr == cls)) * (target / n) for cls in classes}
    test_c = {cls: 0.0 for cls in classes}
    chosen: list = []
    test_n = 0
    remaining = list(order)
    while remaining and test_n < target:
        best = None
        best_score = None
        for group in remaining:
            gsize = len(index[group])
            if test_n > 0 and test_n >= target * 0.75 and test_n + gsize > target * 1.25:
                continue
            if y_arr is None:
                best = group
                break
            added = {cls: float(np.sum(y_arr[index[group]] == cls)) for cls in classes}
            score = 0.0
            for cls in classes:
                score += abs(test_c[cls] - target_c[cls]) - abs(test_c[cls] + added[cls] - target_c[cls])
            score -= abs((test_n + gsize) - target) / n
            if best_score is None or score > best_score:
                best_score = score
                best = group
        if best is None:
            break
        remaining.remove(best)
        chosen.append(best)
        test_n += len(index[best])
        for cls in classes:
            test_c[cls] += float(np.sum(y_arr[index[best]] == cls))
    if not chosen:
        smallest = min(order, key=lambda group: len(index[group]))
        chosen = [smallest]
    test_idx = np.concatenate([index[group] for group in chosen])
    train_mask = np.ones(n, dtype=bool)
    train_mask[test_idx] = False
    train_idx = np.flatnonzero(train_mask)
    if len(train_idx) == 0 or len(test_idx) == 0:
        raise ValueError("group split left one side empty")
    return np.asarray(train_idx, dtype=int), np.asarray(test_idx, dtype=int)


def group_kfold(groups, k, seed, y=None):
    """Yield (train_idx, val_idx). No group appears in more than one fold."""
    groups = np.asarray(groups)
    k = int(k)
    if k < 2:
        raise ValueError("k must be at least 2")
    uniq, index = _group_index(groups)
    if k > len(uniq):
        raise ValueError("k cannot exceed the number of groups")
    rng = np.random.RandomState(int(seed))
    order = uniq.copy()
    rng.shuffle(order)
    folds: list[list] = [[] for _ in range(k)]
    if y is None:
        for i, group in enumerate(order):
            folds[i % k].append(group)
    else:
        y_arr = np.asarray(list(y))
        classes = list(np.unique(y_arr))
        majority = {}
        for group in order:
            vals, counts = np.unique(y_arr[index[group]], return_counts=True)
            majority[group] = vals[int(np.argmax(counts))]
        tally = [{cls: 0 for cls in classes} for _ in range(k)]
        for group in order:
            cls = majority[group]
            fold = min(range(k), key=lambda i: (tally[i][cls], sum(tally[i].values())))
            folds[fold].append(group)
            tally[fold][cls] += len(index[group])
    pairs = []
    for i in range(k):
        if not folds[i]:
            continue
        val = np.concatenate([index[group] for group in folds[i]])
        train_groups = [group for j, fold in enumerate(folds) if j != i for group in fold]
        if not train_groups:
            continue
        train = np.concatenate([index[group] for group in train_groups])
        pairs.append((np.asarray(train, dtype=int), np.asarray(val, dtype=int)))
    return pairs


def filled_rainfall_values(values, min_share: float | None = None) -> list[float]:
    """Values repeated on more than `min_share` of the finite training rows."""
    share = config.FILLED_VALUE_MIN_SHARE if min_share is None else float(min_share)
    series = pd.to_numeric(pd.Series(values), errors="coerce").dropna()
    if series.empty:
        return []
    counts = series.value_counts()
    flagged = counts[counts > share * len(series)]
    return [float(value) for value in flagged.index]


def _rain_columns(frame: pd.DataFrame) -> list[str]:
    return [name for name in ("rainfall", "rainfall_file") if name in frame.columns]


def blank_physical_extremes(frame: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Set rainfall above 2,000 mm and elevation below -20 m to NaN. Keep the row."""
    out = frame.copy()
    touched = 0
    for name in _rain_columns(out):
        rain = pd.to_numeric(out[name], errors="coerce")
        bad = rain > config.FEATURE_RANGES["rainfall"][1]
        touched += int(bad.fillna(False).sum())
        out.loc[bad.fillna(False), name] = np.nan
    if "elevation" in out.columns:
        elev = pd.to_numeric(out["elevation"], errors="coerce")
        bad = elev < config.FEATURE_RANGES["elevation"][0]
        touched += int(bad.fillna(False).sum())
        out.loc[bad.fillna(False), "elevation"] = np.nan
    return out, touched


def apply_measurement_mask(frame: pd.DataFrame, filled_rainfall) -> pd.DataFrame:
    """Blank untrusted rainfall and low elevation, and add missingness indicators.

    `filled_rainfall` must be computed on the training split only.
    """
    out = frame.copy()
    rain_names = _rain_columns(out)
    if rain_names:
        missing = pd.Series(False, index=out.index)
        for name in rain_names:
            rain = pd.to_numeric(out[name], errors="coerce")
            col_missing = rain.isna() | (rain > config.FEATURE_RANGES["rainfall"][1])
            if filled_rainfall:
                col_missing = col_missing | rain.isin(list(filled_rainfall))
            missing = missing | col_missing.fillna(False)
            out.loc[col_missing.fillna(False), name] = np.nan
        out["rainfall_missing"] = missing.astype(float)
    if "elevation" in out.columns:
        elev = pd.to_numeric(out["elevation"], errors="coerce")
        missing = elev.isna() | (elev < config.FEATURE_RANGES["elevation"][0])
        out["elevation_missing"] = missing.astype(float)
        out.loc[missing, "elevation"] = np.nan
    if "rain_missing" not in out.columns and "rain_annual_mm" in out.columns:
        out["rain_missing"] = pd.to_numeric(out["rain_annual_mm"], errors="coerce").isna().astype(float)
    if "pop_missing" not in out.columns and "pop_density_point" in out.columns:
        out["pop_missing"] = pd.to_numeric(out["pop_density_point"], errors="coerce").isna().astype(float)
    if (
        "pop_year_clipped" not in out.columns
        and "pop_density_point" in out.columns
        and "year" in out.columns
    ):
        year = pd.to_numeric(out["year"], errors="coerce")
        out["pop_year_clipped"] = ((year < 2000) | (year > 2020)).astype(float)
    return out


def continent_names(lat, lon) -> np.ndarray:
    """Coarse continent from latitude/longitude boxes. First match wins.

    This is not a coastline. Order: Antarctica (lat < -60); the Americas
    (lon < -30, North if lat >= 13 else South); Africa (-35..37, -20..52);
    Europe (lat >= 36, -25..40); Oceania (lat <= 0 and lon >= 110); else Asia.
    """
    lat = np.asarray(lat, dtype=np.float64)
    lon = np.asarray(lon, dtype=np.float64)
    names = np.array(["Asia"] * len(lat), dtype=object)
    names[lat < -60] = "Antarctica"
    americas = (lon < -30) & (lat >= -60)
    names[americas & (lat >= 13)] = "North America"
    names[americas & (lat < 13)] = "South America"
    africa = (lat >= -35) & (lat <= 37) & (lon >= -20) & (lon <= 52)
    names[africa & (names == "Asia")] = "Africa"
    europe = (lat >= 36) & (lon >= -25) & (lon <= 40)
    names[europe & (names == "Asia")] = "Europe"
    oceania = (lat <= 0) & (lon >= 110)
    names[oceania & (names == "Asia")] = "Oceania"
    return names


def split_positions(frame: pd.DataFrame, split: str, seed: int, test_size: float | None = None, cell_deg: float | None = None):
    """Row positions for the holdout. Spatial blocks stay intact.

    When the label is still a score, group assignment uses quantiles of that
    score as a balance key only. Callers that report classes must refit the
    cuts on the training rows.
    """
    test_size = config.TEST_SIZE if test_size is None else float(test_size)
    if split == "spatial" and {"latitude", "longitude"} <= set(frame.columns):
        cell = config.GRID_CELL_DEG if cell_deg is None else float(cell_deg)
        groups = grid_cell_ids(frame["latitude"], frame["longitude"], cell)
        if config.TARGET in frame.columns and frame[config.TARGET].notna().any():
            guide = frame[config.TARGET].to_numpy()
        elif "flood_hazard_score" in frame.columns:
            guide, _ignored, _cuts = bin_score_train_quantiles(
                frame["flood_hazard_score"], frame["flood_hazard_score"]
            )
        else:
            guide = None
        return group_train_test_split(groups, test_size, seed, y=guide)
    if config.TARGET in frame.columns and frame[config.TARGET].notna().any():
        return stratified_indices(frame[config.TARGET].to_numpy(), test_size, seed)
    rng = np.random.RandomState(int(seed))
    order = np.arange(len(frame))
    rng.shuffle(order)
    n_test = int(round(len(frame) * test_size))
    n_test = min(max(n_test, 1), len(frame) - 1)
    return order[n_test:], order[:n_test]


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
