"""Train the self-implemented forest, score the holdout once, and save artifacts."""

from __future__ import annotations

import json
import math
import sys
import time
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
from flood_rf.data_loader import load_processed, model_feature_columns, source_flag
from flood_rf.explain import permutation_importance, ranked_importance
from flood_rf.features import FeatureEngineer
from flood_rf.metrics import classification_report, evaluate
from flood_rf.persistence import FloodModel, save_model
from flood_rf.preprocessing import (
    Preprocessor,
    apply_measurement_mask,
    bin_score_train_quantiles,
    blank_physical_extremes,
    drop_duplicate_rows,
    encode_labels,
    filled_rainfall_values,
    grid_cell_ids,
    group_kfold,
    remove_invalid_rows,
    split_positions,
    stratified_train_test_split,
)
from flood_rf.random_forest import RandomForestClassifier
from flood_rf.visualize import (
    plot_class_distribution,
    plot_confusion_matrix,
    plot_depth_curve,
    plot_feature_importance,
    plot_learning_curve,
)


def _jsonable(value):
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        number = float(value)
        return number if math.isfinite(number) else None
    if value is None or isinstance(value, str):
        return value
    raise TypeError(type(value))


def _canonicalize(series: pd.Series) -> pd.Series:
    def one(value):
        if pd.isna(value):
            return np.nan
        key = str(value).strip().upper()
        return config.LABEL_SYNONYMS.get(key, key)

    return series.map(one)


def _coerce(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    for column in config.BASE_FEATURES + config.OPTIONAL_FEATURES + config.LABEL_SCORE_COLUMNS:
        if column in out.columns:
            out[column] = pd.to_numeric(out[column], errors="coerce")
    return out


def _encode(y, labels: list[str]) -> np.ndarray:
    encoded, _ = encode_labels(list(y), labels)
    return encoded


def _ranges_and_presets(X_train: pd.DataFrame, y_train) -> tuple[dict, dict, dict]:
    ranges = {}
    defaults = {}
    for column in X_train.columns:
        values = pd.to_numeric(X_train[column], errors="coerce").to_numpy(dtype=np.float64)
        finite = values[np.isfinite(values)]
        if len(finite) == 0:
            continue
        lo = float(finite.min())
        hi = float(finite.max())
        if hi <= lo:
            hi = lo + 1.0
        ranges[column] = [lo, hi]
        defaults[column] = float(np.median(finite))
    y_values = np.asarray(y_train)
    presets = {}
    for label in pd.unique(y_values):
        subset = X_train.iloc[np.flatnonzero(y_values == label)]
        preset = {}
        for column, (lo, hi) in ranges.items():
            values = pd.to_numeric(subset[column], errors="coerce").to_numpy(dtype=np.float64)
            finite = values[np.isfinite(values)]
            median = defaults[column] if len(finite) == 0 else float(np.median(finite))
            preset[column] = float(min(max(median, lo), hi))
        presets[str(label)] = preset
    return ranges, defaults, presets


_BANNED_FEATURES = {
    "total_affected",
    "total_deaths",
    "deaths",
    "disaster_type",
    "occured",
    "occurred",
    "flood_occurrence",
    "flood_hazard_score",
}


def _candidate(params: dict, n_estimators: int, seed: int, n_classes: int) -> RandomForestClassifier:
    return RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=params["max_depth"],
        min_samples_split=config.RF_DEFAULTS["min_samples_split"],
        min_samples_leaf=params["min_samples_leaf"],
        max_features=params["max_features"],
        criterion=config.RF_DEFAULTS["criterion"],
        bootstrap=True,
        class_weight=params.get("class_weight", config.RF_DEFAULTS["class_weight"]),
        oob_score=False,
        n_jobs=1,
        random_state=seed,
    )


def _fold_macro_f1(X_train, y_train, folds, params, n_estimators, seed, labels, clip_outliers) -> float:
    """Mean macro-F1 across training folds. Each fold fits its own imputer."""
    engineer = FeatureEngineer()
    scores = []
    for train_idx, val_idx in folds:
        y_tr = y_train.iloc[train_idx]
        y_val = y_train.iloc[val_idx]
        if len(val_idx) < 5 or y_tr.nunique() < 2:
            continue
        pre = Preprocessor(clip_outliers=clip_outliers).fit(X_train.iloc[train_idx])
        x_tr = engineer.transform(pre.transform(X_train.iloc[train_idx]))
        x_val = engineer.transform(pre.transform(X_train.iloc[val_idx]))
        names = list(x_tr.columns)
        forest = _candidate(params, n_estimators, seed, len(labels))
        forest.fit(x_tr[names].to_numpy(dtype=np.float64), _encode(y_tr, labels), n_classes=len(labels))
        predicted = [labels[i] for i in forest.predict(x_val[names].to_numpy(dtype=np.float64))]
        scores.append(evaluate(y_val, predicted, labels)["f1_macro"])
    if not scores:
        return -1.0
    return float(np.mean(scores))


def _guess_labels(y_train_enc, n_test: int, labels: list[str], seed: int) -> list[str]:
    """One draw from the training class prior. Not used to choose the model."""
    counts = np.bincount(y_train_enc, minlength=len(labels)).astype(np.float64)
    prior = counts / counts.sum()
    drawn = np.random.RandomState(int(seed)).choice(len(labels), size=n_test, p=prior)
    return [labels[i] for i in drawn]


def cleaned_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Rows that reach the splitter. Used so a custom holdout indexes the same table."""
    frame = _coerce(df).reset_index(drop=True)
    score_col = None
    if config.TARGET not in frame.columns or not frame[config.TARGET].notna().any():
        for candidate in ("flood_hazard_score", "historical_flood"):
            if candidate in frame.columns:
                score_col = candidate
                break
    else:
        frame[config.TARGET] = _canonicalize(frame[config.TARGET])
    ranges = dict(config.FEATURE_RANGES)
    if score_col:
        ranges.pop(score_col, None)
    frame, _n_blanked = blank_physical_extremes(frame)
    frame, _n_invalid = remove_invalid_rows(frame, ranges)
    duplicate_on = [column for column in frame.columns if column not in {"row_id", "pop_source_year"}]
    frame, _n_dupes = drop_duplicate_rows(frame, subset=duplicate_on or None)
    if config.TARGET in frame.columns and score_col is None:
        frame = frame[frame[config.TARGET].isin(config.CLASS_LABELS)].reset_index(drop=True)
    return frame


def train_dataframe(
    df: pd.DataFrame,
    search: bool = True,
    explain: bool = True,
    n_estimators_final: int | None = None,
    random_state: int | None = None,
    verbose: bool = False,
    clip_outliers: bool = False,
    split: str = "random",
    feature_keep: list[str] | None = None,
    fixed_params: dict | None = None,
    split_seed: int | None = None,
    cell_deg: float | None = None,
    positions: tuple | None = None,
) -> dict:
    """Fit on a dataframe and return the model plus holdout metrics. Does not write files.

    `split="random"` is the old row split (optimistic when neighbors share a cell).
    `split="spatial"` holds out whole grid cells. The test rows are scored once.
    `split_seed` moves the spatial blocks without changing the forest seed.
    """
    started = time.perf_counter()
    seed = config.RANDOM_STATE if random_state is None else int(random_state)
    holdout_seed = seed if split_seed is None else int(split_seed)
    cell = config.GRID_CELL_DEG if cell_deg is None else float(cell_deg)
    n_final = config.RF_DEFAULTS["n_estimators"] if n_estimators_final is None else int(n_estimators_final)
    frame = _coerce(df).reset_index(drop=True)
    source = source_flag(frame)
    score_col = None
    label_rule = "risk_class column on the processed table"
    if config.TARGET not in frame.columns or not frame[config.TARGET].notna().any():
        for candidate in ("flood_hazard_score", "historical_flood"):
            if candidate in frame.columns:
                score_col = candidate
                break
        if score_col is None:
            raise ValueError("Need risk_class or a hazard score column")
        label_rule = (
            f"Quantiles {config.TRAIN_SCORE_QUANTILES} of {score_col} fit on the training split only"
        )
    else:
        frame[config.TARGET] = _canonicalize(frame[config.TARGET])

    ranges = dict(config.FEATURE_RANGES)
    if score_col:
        ranges.pop(score_col, None)
    frame, _n_blanked = blank_physical_extremes(frame)
    frame, n_invalid = remove_invalid_rows(frame, ranges)
    duplicate_on = [column for column in frame.columns if column not in {"row_id", "pop_source_year"}]
    frame, n_dupes = drop_duplicate_rows(frame, subset=duplicate_on or None)
    if config.TARGET in frame.columns and not score_col:
        frame = frame[frame[config.TARGET].isin(config.CLASS_LABELS)].reset_index(drop=True)

    if positions is None:
        train_pos, test_pos = split_positions(frame, split, holdout_seed, cell_deg=cell)
    else:
        train_pos = np.asarray(positions[0], dtype=int)
        test_pos = np.asarray(positions[1], dtype=int)
    train_df = frame.iloc[train_pos].reset_index(drop=True)
    test_df = frame.iloc[test_pos].reset_index(drop=True)
    if score_col:
        train_labels, test_labels, cuts = bin_score_train_quantiles(
            train_df[score_col], test_df[score_col]
        )
        label_rule += f"; training cuts at {tuple(round(float(c), 4) for c in cuts)}"
        train_df[config.TARGET] = train_labels
        test_df[config.TARGET] = test_labels
        keep_train = train_df[config.TARGET].notna()
        keep_test = test_df[config.TARGET].notna()
        train_df = train_df.loc[keep_train].reset_index(drop=True)
        test_df = test_df.loc[keep_test].reset_index(drop=True)
    rain_col = next((name for name in ("rainfall_file", "rainfall") if name in train_df.columns), None)
    filled = filled_rainfall_values(train_df[rain_col]) if rain_col else []
    rain_before = pd.to_numeric(train_df[rain_col], errors="coerce") if rain_col else None
    filled_share = 0.0
    if rain_before is not None and len(rain_before):
        filled_share = float(rain_before.isin(filled).mean()) if filled else 0.0
    train_df = apply_measurement_mask(train_df, filled)
    test_df = apply_measurement_mask(test_df, filled)
    columns = model_feature_columns(train_df, drop=[score_col] if score_col else None)
    if feature_keep is not None:
        columns = [column for column in feature_keep if column in train_df.columns]
    if not columns:
        raise ValueError("No usable feature columns")
    leaked = _BANNED_FEATURES.intersection(columns)
    if leaked:
        raise RuntimeError(f"Refusing to train on label columns: {sorted(leaked)}")
    X_train = train_df[columns].copy()
    X_test = test_df[columns].copy()
    y_train = train_df[config.TARGET].copy()
    y_test = test_df[config.TARGET].copy()
    search_groups = None
    if split == "spatial" and {"latitude", "longitude"} <= set(train_df.columns):
        search_groups = grid_cell_ids(train_df["latitude"], train_df["longitude"], cell)
    if len(X_test) < 5:
        raise ValueError("Holdout split has fewer than 5 rows")
    labels = [label for label in config.CLASS_LABELS if label in set(y_train) | set(y_test)]
    if len(labels) < 2:
        raise ValueError("Need at least two classes after cleaning")

    engineer = FeatureEngineer()
    search_log = []
    weight_log = []
    chosen = {
        "max_depth": config.RF_DEFAULTS["max_depth"],
        "min_samples_leaf": config.RF_DEFAULTS["min_samples_leaf"],
        "max_features": config.RF_DEFAULTS["max_features"],
        "class_weight": config.RF_DEFAULTS["class_weight"],
    }
    if fixed_params:
        chosen.update(fixed_params)
        search = False
    elif search and len(X_train) >= 50 and len(set(y_train)) >= 2:
        if search_groups is not None:
            try:
                folds = group_kfold(search_groups, config.GROUP_FOLDS, seed + 1, y=y_train.to_numpy())
            except ValueError:
                folds = None
        else:
            folds = None
        if not folds:
            x_tr, x_val, y_tr, y_val = stratified_train_test_split(
                X_train, y_train, test_size=config.VALIDATION_FRACTION, random_state=seed + 1
            )
            folds = [(np.arange(len(x_tr)), np.arange(len(x_tr), len(x_tr) + len(x_val)))]
            X_train = pd.concat([x_tr, x_val], ignore_index=True)
            y_train = pd.concat([y_tr, y_val], ignore_index=True)
        best_score = -1.0
        for params in config.SEARCH_GRID:
            if verbose:
                print(f"search {params}", flush=True)
            score = _fold_macro_f1(
                X_train, y_train, folds, params, config.SEARCH_N_ESTIMATORS, seed, labels, clip_outliers
            )
            row = dict(params)
            row["n_estimators"] = config.SEARCH_N_ESTIMATORS
            row["class_weight"] = config.RF_DEFAULTS["class_weight"]
            row["validation_macro_f1"] = score
            row["objective"] = "macro_f1"
            search_log.append(row)
            if score > best_score:
                best_score = score
                chosen = dict(params)
                chosen["class_weight"] = config.RF_DEFAULTS["class_weight"]
        if search_groups is not None:
            best_weight_score = -1.0
            for n_estimators in config.SEARCH_N_ESTIMATOR_CHOICES:
                for weight in config.SEARCH_CLASS_WEIGHTS:
                    trial = dict(chosen)
                    trial["class_weight"] = weight
                    if verbose:
                        print(f"search n_estimators={n_estimators} class_weight={weight}", flush=True)
                    score = _fold_macro_f1(
                        X_train, y_train, folds, trial, n_estimators, seed, labels, clip_outliers
                    )
                    weight_log.append(
                        {
                            "n_estimators": n_estimators,
                            "class_weight": weight,
                            "validation_macro_f1": score,
                        }
                    )
                    if score > best_weight_score:
                        best_weight_score = score
                        n_final = n_estimators
                        chosen["class_weight"] = weight
    else:
        search = False

    preprocess = Preprocessor(clip_outliers=clip_outliers).fit(X_train)
    X_train_f = engineer.transform(preprocess.transform(X_train))
    X_test_f = engineer.transform(preprocess.transform(X_test))
    feature_names = list(X_train_f.columns)
    X_train_m = X_train_f[feature_names].to_numpy(dtype=np.float64)
    X_test_m = X_test_f[feature_names].to_numpy(dtype=np.float64)
    y_train_enc = _encode(y_train, labels)
    y_test_enc = _encode(y_test, labels)

    forest = RandomForestClassifier(
        n_estimators=n_final,
        max_depth=chosen["max_depth"],
        min_samples_split=config.RF_DEFAULTS["min_samples_split"],
        min_samples_leaf=chosen["min_samples_leaf"],
        max_features=chosen["max_features"],
        criterion=config.RF_DEFAULTS["criterion"],
        bootstrap=True,
        class_weight=chosen.get("class_weight", config.RF_DEFAULTS["class_weight"]),
        oob_score=True,
        n_jobs=1,
        random_state=seed,
        verbose=verbose,
    )
    forest.fit(X_train_m, y_train_enc, n_classes=len(labels))
    train_pred = [labels[i] for i in forest.predict(X_train_m)]
    test_pred = [labels[i] for i in forest.predict(X_test_m)]
    train_stats = evaluate(y_train, train_pred, labels)
    test_stats = evaluate(y_test, test_pred, labels)
    majority = int(np.bincount(y_train_enc, minlength=len(labels)).argmax())
    baseline = float(np.mean(y_test_enc == majority))
    majority_pred = [labels[majority]] * len(y_test)
    guess_pred = _guess_labels(y_train_enc, len(y_test), labels, seed)
    perm = (
        permutation_importance(forest, X_test_m, y_test_enc, feature_names, random_state=seed)
        if explain
        else []
    )
    input_ranges, defaults, presets = _ranges_and_presets(X_train, y_train)
    distribution = {label: int(sum(value == label for value in list(y_train) + list(y_test))) for label in labels}
    hyperparameters = {
        "n_estimators": n_final,
        "max_depth": chosen["max_depth"],
        "min_samples_split": config.RF_DEFAULTS["min_samples_split"],
        "min_samples_leaf": chosen["min_samples_leaf"],
        "max_features": chosen["max_features"],
        "criterion": config.RF_DEFAULTS["criterion"],
        "class_weight": chosen.get("class_weight", config.RF_DEFAULTS["class_weight"]),
        "bootstrap": True,
        "random_state": seed,
        "search_n_estimators": config.SEARCH_N_ESTIMATORS if search else None,
    }
    metrics = {
        "data_source": source,
        "warning": config.warning_for(source),
        "disclaimer": config.DISCLAIMER,
        "label_rule": label_rule,
        "n_rows": int(len(y_train) + len(y_test)),
        "n_train": int(len(y_train)),
        "n_test": int(len(y_test)),
        "dropped_invalid": int(n_invalid),
        "dropped_duplicates": int(n_dupes),
        "features": feature_names,
        "base_features": list(X_train.columns),
        "labels": labels,
        "class_distribution": distribution,
        "hyperparameters": hyperparameters,
        "search": search_log,
        "class_weight_validation": weight_log,
        "objective": "macro_f1",
        "objective_reason": (
            "Macro F1 weights the three classes equally. "
            "Quadratic weighted kappa is reported and is not used to pick the model."
        ),
        "split": split,
        "train": train_stats,
        "test": test_stats,
        "baseline_accuracy": baseline,
        "baselines": {
            "majority": evaluate(y_test, majority_pred, labels),
            "stratified_random_guess": evaluate(y_test, guess_pred, labels),
        },
        "filled_rainfall": filled,
        "rainfall_filled_share_train": filled_share,
        "oob_score": forest.oob_score_,
        "overfit_gap_accuracy": float(train_stats["accuracy"] - test_stats["accuracy"]),
        "train_seconds": float(time.perf_counter() - started),
        "test_report": classification_report(y_test, test_pred, labels),
        "learning_curve": [{"n_estimators": int(n), "oob_accuracy": float(score)} for n, score in forest.learning_curve_],
    }
    extra = {
        "input_ranges": input_ranges,
        "defaults": defaults,
        "presets": presets,
        "data_source": source,
        "label_rule": label_rule,
        "filled_rainfall": filled,
    }
    user_columns = [column for column in X_train.columns if column not in config.INDICATOR_FEATURES]
    model = FloodModel(forest, preprocess, feature_names, labels, user_columns, extra)
    return {
        "model": model,
        "metrics": _jsonable(metrics),
        "impurity_importance": ranked_importance(feature_names, forest.feature_importances_),
        "permutation_importance": perm,
        "search": search_log,
        "positions": (train_pos, test_pos),
        "y_test": list(y_test),
        "test_pred": list(test_pred),
    }


def _print_summary(metrics: dict) -> None:
    print("=" * 64)
    print("Urban flood risk - self-implemented random forest")
    print(f"DATA SOURCE: {metrics['data_source']}")
    if metrics["data_source"] != "REAL":
        print(metrics["warning"])
    print(f"rows={metrics['n_rows']} train={metrics['n_train']} test={metrics['n_test']}")
    print(f"features: {', '.join(metrics['features'])}")
    print(f"hyperparameters: {metrics['hyperparameters']}")
    print(
        f"TRAIN accuracy={metrics['train']['accuracy']:.3f} "
        f"macro-F1={metrics['train']['f1_macro']:.3f}"
    )
    print(
        f"TEST ({metrics.get('split', 'random')}) accuracy={metrics['test']['accuracy']:.3f} "
        f"macro-F1={metrics['test']['f1_macro']:.3f} "
        f"QWK={metrics['test'].get('quadratic_weighted_kappa', float('nan')):.3f} "
        f"precision={metrics['test']['precision_macro']:.3f} "
        f"recall={metrics['test']['recall_macro']:.3f}"
    )
    print(f"majority baseline accuracy={metrics['baseline_accuracy']:.3f}")
    print(f"OOB accuracy={metrics['oob_score']}")
    print(f"train-test accuracy gap={metrics['overfit_gap_accuracy']:.3f}")
    print(f"fit+search seconds={metrics['train_seconds']:.1f}")
    print(metrics["test_report"])
    print(config.DISCLAIMER)


def _public_scores(metrics: dict) -> dict:
    test = metrics["test"]
    return {
        "accuracy": test["accuracy"],
        "f1_macro": test["f1_macro"],
        "quadratic_weighted_kappa": test["quadratic_weighted_kappa"],
        "adjacent_accuracy": test["adjacent_accuracy"],
        "mean_absolute_class_error": test["mean_absolute_class_error"],
        "severe_error_rate": test["severe_error_rate"],
        "baseline_accuracy": metrics["baseline_accuracy"],
        "majority_f1_macro": metrics["baselines"]["majority"]["f1_macro"],
        "majority_qwk": metrics["baselines"]["majority"]["quadratic_weighted_kappa"],
        "n_train": metrics["n_train"],
        "n_test": metrics["n_test"],
        "features": metrics["base_features"],
    }


def _fixed(metrics: dict) -> tuple[dict, int]:
    params = metrics["hyperparameters"]
    fixed = {
        "max_depth": params["max_depth"],
        "min_samples_leaf": params["min_samples_leaf"],
        "max_features": params["max_features"],
        "class_weight": params["class_weight"],
    }
    return fixed, int(params["n_estimators"])


def main() -> None:
    if not config.DATASET_CSV.exists():
        raise SystemExit("Missing data/processed/dataset.csv. Run: python scripts/build_dataset.py")
    frame = load_processed()
    use_spatial = {"latitude", "longitude"} <= set(frame.columns)
    headline_split = "spatial" if use_spatial else "random"
    external_names = [name for name in config.RAIN_EXT + config.POP_EXT if name in frame.columns]
    headline_keep = None
    if external_names and use_spatial:
        from external_eval import full_feature_set

        headline_keep = full_feature_set(frame.columns)
    result = train_dataframe(
        frame,
        search=True,
        explain=True,
        verbose=True,
        split=headline_split,
        feature_keep=headline_keep,
    )
    fixed, n_estimators = _fixed(result["metrics"])
    if use_spatial:
        print("random split (comparison, not used to choose the model)", flush=True)
        random_run = train_dataframe(
            frame,
            search=False,
            explain=False,
            split="random",
            fixed_params=fixed,
            n_estimators_final=n_estimators,
        )
        result["metrics"]["random_split"] = {
            "note": "optimistic, leakage-prone",
            **_public_scores(random_run["metrics"]),
        }
        rain = "rainfall_file" if "rainfall_file" in frame.columns else "rainfall"
        sets = [
            ("rainfall_elevation_slope", [rain, "elevation", "slope"]),
            ("plus_lat_lon", [rain, "elevation", "slope", "latitude", "longitude"]),
            ("plus_year", [rain, "elevation", "slope", "latitude", "longitude", "year"]),
            (
                "plus_indicators",
                [rain, "elevation", "slope", "latitude", "longitude", "year", "rainfall_missing", "elevation_missing"],
            ),
            ("previous_features", [rain, "elevation", "slope", "latitude", "longitude", "year", "rainfall_missing", "elevation_missing"]),
        ]
        ablation = []
        for name, columns in sets:
            print(f"ablation {name}", flush=True)
            if columns is None:
                scores = _public_scores(result["metrics"])
            else:
                run = train_dataframe(
                    frame,
                    search=False,
                    explain=False,
                    split="spatial",
                    feature_keep=columns,
                    fixed_params=fixed,
                    n_estimators_final=n_estimators,
                )
                scores = _public_scores(run["metrics"])
            ablation.append({"name": name, **scores})
        result["metrics"]["ablation"] = ablation
        ablation_path = ROOT / "reports" / "ablation.csv"
        ablation_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(ablation).to_csv(ablation_path, index=False)
        seed_rows = []
        for seed in config.VARIANCE_SEEDS:
            print(f"seed {seed}", flush=True)
            if seed == result["metrics"]["hyperparameters"]["random_state"]:
                scores = _public_scores(result["metrics"])
            else:
                run = train_dataframe(
                    frame,
                    search=False,
                    explain=False,
                    split="spatial",
                    feature_keep=headline_keep,
                    fixed_params=fixed,
                    n_estimators_final=n_estimators,
                    random_state=seed,
                )
                scores = _public_scores(run["metrics"])
            seed_rows.append({"seed": int(seed), **scores})
        result["metrics"]["seed_runs"] = seed_rows
        result["metrics"]["seed_variance"] = {
            metric: {
                "mean": float(np.mean([row[metric] for row in seed_rows])),
                "std": float(np.std([row[metric] for row in seed_rows], ddof=1)),
            }
            for metric in ("accuracy", "f1_macro", "quadratic_weighted_kappa")
        }
        if external_names:
            from external_eval import evaluate_external, write_csv

            print("external ablation", flush=True)
            external = evaluate_external(frame, result, fixed, n_estimators)
            result["metrics"]["external"] = _jsonable(external)
            write_csv(external, ROOT / "reports" / "ablation_external.csv")
    result["metrics"]["headline"] = headline_split
    if config.DATASET_META.exists():
        meta = json.loads(config.DATASET_META.read_text(encoding="utf-8"))
        rule = meta.get("label_rule")
        if rule:
            trained = result["metrics"]["label_rule"]
            marker = "training cuts at"
            if marker in trained and marker not in rule:
                rule = f"{rule} {trained[trained.index(marker):]}"
            result["metrics"]["label_rule"] = rule
            result["model"].extra["label_rule"] = rule
    save_model(result["model"])
    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    config.METRICS_PATH.write_text(json.dumps(result["metrics"], indent=2), encoding="utf-8")
    importance = {
        "note": config.CAUSATION_NOTE,
        "impurity": result["impurity_importance"],
        "permutation": result["permutation_importance"],
    }
    config.IMPORTANCE_PATH.write_text(json.dumps(_jsonable(importance), indent=2), encoding="utf-8")
    labels = result["metrics"]["labels"]
    matrix = result["metrics"]["test"]["confusion_matrix"]
    plot_confusion_matrix(matrix, labels, config.FIGURES_DIR / "confusion_matrix.png")
    plot_confusion_matrix(matrix, labels, config.FIGURES_DIR / "confusion_matrix_normalized.png", normalize=True)
    impurity = result["impurity_importance"]
    plot_feature_importance(
        [row["feature"] for row in impurity],
        [row["importance"] for row in impurity],
        config.FIGURES_DIR / "feature_importance.png",
        title="Impurity feature importance",
    )
    perm = result["permutation_importance"]
    if perm:
        plot_feature_importance(
            [row["feature"] for row in perm],
            [row["importance"] for row in perm],
            config.FIGURES_DIR / "permutation_importance.png",
            title="Permutation importance (held-out)",
        )
    plot_class_distribution(result["metrics"]["class_distribution"], config.FIGURES_DIR / "class_distribution.png")
    curve = [(point["n_estimators"], point["oob_accuracy"]) for point in result["metrics"]["learning_curve"]]
    plot_learning_curve(curve, config.FIGURES_DIR / "learning_curve.png")
    plot_depth_curve(result["search"], config.FIGURES_DIR / "depth_curve.png")
    _print_summary(result["metrics"])


if __name__ == "__main__":
    main()
