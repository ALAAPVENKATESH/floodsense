"""Same-cell ablations for CHIRPS and population features.

One spatial split is drawn per seed and reused for every feature set.
The forest seed stays at config.RANDOM_STATE. Only the spatial-split seed changes.
Hyperparameters are the ones already chosen on the training folds.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

import config
from flood_rf.metrics import bootstrap_ci
from flood_rf.preprocessing import continent_names
from train import _public_scores, cleaned_frame, train_dataframe

PUBLISHED_HEADLINE = {
    "accuracy": 0.470,
    "f1_macro": 0.459,
    "quadratic_weighted_kappa": 0.242,
}
PREVIOUS_PARAMS = {
    "max_depth": 8,
    "min_samples_leaf": 8,
    "max_features": 0.5,
    "class_weight": "balanced",
}
PREVIOUS_TREES = 40


def _rain(columns) -> str:
    have = set(columns)
    if "rainfall_file" in have:
        return "rainfall_file"
    return "rainfall"


def feature_sets(columns) -> list[tuple[str, list[str]]]:
    have = set(columns)
    rain = _rain(have)
    # NOTE: rainfall_file is intentionally excluded here — it is the direct label variable.
    # rain_missing / pop_missing / pop_year_clipped are excluded because rain_missing=1 for
    # every LOW row, making it a perfect class proxy (data leakage).
    physical = [name for name in ("elevation", "slope", "rainfall_missing", "elevation_missing") if name in have]
    loc = physical + [name for name in ("latitude", "longitude") if name in have]
    chirps = [name for name in list(config.RAIN_EXT) if name in have]  # rain_missing removed
    pop = [name for name in list(config.POP_EXT) if name in have]  # pop_missing/pop_year_clipped removed
    return [
        ("physical", physical),
        ("plus_lat_lon", loc),
        ("plus_chirps", loc + chirps),
        ("plus_population", loc + pop),
        ("full", loc + chirps + pop),
        ("no_lat_lon", chirps + pop),
    ]


def full_feature_set(columns) -> list[str]:
    return feature_sets(columns)[4][1]


def _fit(frame, columns, fixed, n_estimators, split_seed, positions, cell_deg=None):
    return train_dataframe(
        frame,
        search=False,
        explain=False,
        split="spatial",
        feature_keep=columns,
        fixed_params=fixed,
        n_estimators_final=n_estimators,
        random_state=config.RANDOM_STATE,
        split_seed=int(split_seed),
        cell_deg=cell_deg,
        positions=positions,
    )


def run_sets(frame, sets, fixed, n_estimators, seeds, population: str) -> list[dict]:
    """Score every feature set on the same cells. `seeds` vary the spatial split only."""
    rows = []
    for seed in seeds:
        positions = None
        for name, columns in sets:
            if not columns:
                continue
            result = _fit(frame, columns, fixed, n_estimators, seed, positions)
            if positions is None:
                positions = result["positions"]
            scores = _public_scores(result["metrics"])
            rows.append(
                {
                    "population": population,
                    "feature_set": name,
                    "seed": int(seed),
                    "positions": positions,
                    **scores,
                }
            )
    return rows


def _mean_rows(rows: list[dict]) -> list[dict]:
    frame = pd.DataFrame(rows)
    summary = []
    metrics = ("accuracy", "f1_macro", "quadratic_weighted_kappa", "adjacent_accuracy", "severe_error_rate")
    for (population, feature_set), block in frame.groupby(["population", "feature_set"], sort=False):
        item = {
            "population": population,
            "feature_set": feature_set,
            "seed": "mean",
            "n_train": int(block["n_train"].iloc[0]),
            "n_test": int(block["n_test"].mean()),
        }
        for metric in metrics:
            item[metric] = float(block[metric].mean())
            item[f"{metric}_std"] = float(block[metric].std(ddof=1)) if len(block) > 1 else 0.0
        summary.append(item)
    return summary


def _paired(rows: list[dict], population: str) -> dict:
    subset = [row for row in rows if row["population"] == population]
    if not subset:
        return {"per_seed": []}
    frame = pd.DataFrame(subset)
    per_seed = []
    for seed, block in frame.groupby("seed"):
        ref = block.loc[block["feature_set"] == "plus_lat_lon"]
        full = block.loc[block["feature_set"] == "full"]
        bare = block.loc[block["feature_set"] == "no_lat_lon"]
        if ref.empty or full.empty:
            continue
        item = {"seed": int(seed)}
        for metric in ("accuracy", "f1_macro", "quadratic_weighted_kappa"):
            item[f"full_minus_latlon_{metric}"] = float(full.iloc[0][metric] - ref.iloc[0][metric])
            if not bare.empty:
                item[f"noloc_minus_latlon_{metric}"] = float(bare.iloc[0][metric] - ref.iloc[0][metric])
        per_seed.append(item)
    out = {"per_seed": per_seed}
    if per_seed:
        table = pd.DataFrame(per_seed)
        for column in table.columns:
            if column == "seed":
                continue
            out[f"{column}_mean"] = float(table[column].mean())
            out[f"{column}_std"] = float(table[column].std(ddof=1)) if len(table) > 1 else 0.0
    return out


def _covered_mask(frame: pd.DataFrame) -> pd.Series:
    year = pd.to_numeric(frame["year"], errors="coerce")
    rain_ok = frame["rain_missing"].fillna(1) == 0 if "rain_missing" in frame.columns else False
    pop_ok = frame["pop_missing"].fillna(1) == 0 if "pop_missing" in frame.columns else False
    return (year >= 2000) & rain_ok & pop_ok


def _continents(frame, columns, fixed, n_estimators) -> list[dict]:
    frame = cleaned_frame(frame)
    names = continent_names(frame["latitude"], frame["longitude"])
    rows = []
    for name in sorted(set(names.tolist())):
        test_idx = np.flatnonzero(names == name)
        train_idx = np.flatnonzero(names != name)
        if len(test_idx) < 5 or len(train_idx) < 20:
            rows.append({"continent": name, "n_test": int(len(test_idx)), "skipped": "too few rows"})
            continue
        try:
            result = _fit(frame, columns, fixed, n_estimators, config.RANDOM_STATE, (train_idx, test_idx))
        except ValueError as exc:
            rows.append({"continent": name, "n_test": int(len(test_idx)), "skipped": str(exc)})
            continue
        scores = _public_scores(result["metrics"])
        rows.append({"continent": name, "n_test": scores["n_test"], **{key: scores[key] for key in ("accuracy", "f1_macro", "quadratic_weighted_kappa", "adjacent_accuracy", "severe_error_rate")}})
    return rows


def evaluate_external(frame: pd.DataFrame, headline: dict, fixed: dict, n_estimators: int) -> dict:
    sets = feature_sets(frame.columns)
    seeds = list(config.VARIANCE_SEEDS)
    rows_a = run_sets(frame, sets, fixed, n_estimators, seeds, "A_all")
    mask = _covered_mask(frame)
    covered = frame.loc[mask].reset_index(drop=True)
    rows_b = []
    if len(covered) >= 40 and covered["latitude"].nunique() > 1:
        rows_b = run_sets(covered, sets, fixed, n_estimators, seeds, "B_covered")
    rows = rows_a + rows_b
    full_columns = full_feature_set(frame.columns)
    wide = _fit(frame, full_columns, fixed, n_estimators, config.RANDOM_STATE, None, cell_deg=15.0)
    wide_scores = _public_scores(wide["metrics"])
    continents = _continents(frame, full_columns, fixed, n_estimators)
    scored = [row for row in continents if "accuracy" in row]
    weighted = None
    if scored:
        weights = np.array([row["n_test"] for row in scored], dtype=float)
        weighted = float(np.average([row["accuracy"] for row in scored], weights=weights))
    previous_columns = [
        name
        for name in (
            _rain(frame.columns),
            "elevation",
            "slope",
            "latitude",
            "longitude",
            "year",
            "rainfall_missing",
            "elevation_missing",
        )
        if name in frame.columns or name in {"rainfall_missing", "elevation_missing"}
    ]
    previous = _fit(frame, previous_columns, PREVIOUS_PARAMS, PREVIOUS_TREES, config.RANDOM_STATE, None)
    previous_scores = _public_scores(previous["metrics"])
    test = headline["metrics"]["test"]
    headline_acc = float(test["accuracy"])
    payload = {
        "note": (
            "Feature sets in one population share the spatial cells. "
            "Seeds change the spatial split only; the forest seed stays 42. "
            "Population B is year >= 2000 with both external sources present. "
            "Hyperparameters were chosen on population A training folds and frozen. "
            "WorldPop was not downloaded, so population measurements are missing on every row. "
            "pop_year_clipped only marks years outside 2000-2020."
        ),
        "published_previous": PUBLISHED_HEADLINE,
        "previous_same_cells": previous_scores,
        "n_A": int(len(frame)),
        "n_B": int(mask.sum()),
        "ablation_mean": _mean_rows(rows),
        "paired_A": _paired(rows, "A_all"),
        "paired_B": _paired(rows, "B_covered"),
        "bootstrap": bootstrap_ci(headline["y_test"], headline["test_pred"], headline["metrics"]["labels"]),
        "cell_15": wide_scores,
        "cell_15_accuracy_drop": float(headline_acc - wide_scores["accuracy"]),
        "continents": continents,
        "continent_weighted_accuracy": weighted,
        "continent_accuracy_drop": None if weighted is None else float(headline_acc - weighted),
        "rows": [{key: value for key, value in row.items() if key != "positions"} for row in rows],
    }
    return payload


def write_csv(payload: dict, path) -> None:
    seed_rows = payload["rows"]
    mean_rows = payload["ablation_mean"]
    table = pd.DataFrame(seed_rows + mean_rows)
    keep = [
        "population",
        "feature_set",
        "seed",
        "accuracy",
        "f1_macro",
        "quadratic_weighted_kappa",
        "adjacent_accuracy",
        "severe_error_rate",
        "accuracy_std",
        "f1_macro_std",
        "quadratic_weighted_kappa_std",
        "adjacent_accuracy_std",
        "severe_error_rate_std",
        "n_train",
        "n_test",
    ]
    for column in keep:
        if column not in table.columns:
            table[column] = np.nan
    path.parent.mkdir(parents=True, exist_ok=True)
    table[keep].to_csv(path, index=False)
