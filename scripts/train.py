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
    bin_score_train_quantiles,
    drop_duplicate_rows,
    encode_labels,
    remove_invalid_rows,
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


def _candidate(params: dict, n_estimators: int, seed: int, n_classes: int) -> RandomForestClassifier:
    return RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=params["max_depth"],
        min_samples_split=config.RF_DEFAULTS["min_samples_split"],
        min_samples_leaf=params["min_samples_leaf"],
        max_features=params["max_features"],
        criterion=config.RF_DEFAULTS["criterion"],
        bootstrap=True,
        class_weight=config.RF_DEFAULTS["class_weight"],
        oob_score=False,
        n_jobs=1,
        random_state=seed,
    )


def train_dataframe(
    df: pd.DataFrame,
    search: bool = True,
    explain: bool = True,
    n_estimators_final: int | None = None,
    random_state: int | None = None,
    verbose: bool = False,
    clip_outliers: bool = False,
) -> dict:
    """Fit on a dataframe and return the model plus holdout metrics. Does not write files."""
    started = time.perf_counter()
    seed = config.RANDOM_STATE if random_state is None else int(random_state)
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
    frame, n_invalid = remove_invalid_rows(frame, ranges)
    frame, n_dupes = drop_duplicate_rows(frame)

    if score_col:
        rng = np.random.RandomState(seed)
        order = np.arange(len(frame))
        rng.shuffle(order)
        n_test = int(round(len(frame) * config.TEST_SIZE))
        n_test = min(max(n_test, 1), len(frame) - 1)
        test_pos = order[:n_test]
        train_pos = order[n_test:]
        train_labels, test_labels, cuts = bin_score_train_quantiles(
            frame.iloc[train_pos][score_col],
            frame.iloc[test_pos][score_col],
        )
        label_rule += f"; training cuts at {tuple(round(c, 4) for c in cuts)}"
        train_df = frame.iloc[train_pos].reset_index(drop=True)
        test_df = frame.iloc[test_pos].reset_index(drop=True)
        train_df[config.TARGET] = train_labels
        test_df[config.TARGET] = test_labels
        train_df = train_df[train_df[config.TARGET].notna()].reset_index(drop=True)
        test_df = test_df[test_df[config.TARGET].notna()].reset_index(drop=True)
        columns = model_feature_columns(train_df, drop=[score_col])
        X_train = train_df[columns].copy()
        X_test = test_df[columns].copy()
        y_train = train_df[config.TARGET].copy()
        y_test = test_df[config.TARGET].copy()
    else:
        frame = frame[frame[config.TARGET].isin(config.CLASS_LABELS)].reset_index(drop=True)
        columns = model_feature_columns(frame)
        if not columns:
            raise ValueError("No usable feature columns")
        X_train, X_test, y_train, y_test = stratified_train_test_split(
            frame[columns],
            frame[config.TARGET],
            test_size=config.TEST_SIZE,
            random_state=seed,
        )
    if len(X_test) < 5:
        raise ValueError("Holdout split has fewer than 5 rows")
    labels = [label for label in config.CLASS_LABELS if label in set(y_train) | set(y_test)]
    if len(labels) < 2:
        raise ValueError("Need at least two classes after cleaning")

    engineer = FeatureEngineer()
    search_log = []
    chosen = {
        "max_depth": config.RF_DEFAULTS["max_depth"],
        "min_samples_leaf": config.RF_DEFAULTS["min_samples_leaf"],
        "max_features": config.RF_DEFAULTS["max_features"],
    }
    if search and len(X_train) >= 50 and len(set(y_train)) >= 2:
        X_tr, X_val, y_tr, y_val = stratified_train_test_split(
            X_train, y_train, test_size=config.VALIDATION_FRACTION, random_state=seed + 1
        )
        pre_search = Preprocessor(clip_outliers=clip_outliers).fit(X_tr)
        X_tr_m = engineer.transform(pre_search.transform(X_tr))
        X_val_m = engineer.transform(pre_search.transform(X_val))
        names = list(X_tr_m.columns)
        y_tr_enc = _encode(y_tr, labels)
        best_score = -1.0
        for params in config.SEARCH_GRID:
            if verbose:
                print(f"search {params}", flush=True)
            forest = _candidate(params, config.SEARCH_N_ESTIMATORS, seed, len(labels))
            forest.fit(X_tr_m[names].to_numpy(dtype=np.float64), y_tr_enc, n_classes=len(labels))
            predicted = [labels[i] for i in forest.predict(X_val_m[names].to_numpy(dtype=np.float64))]
            score = evaluate(y_val, predicted, labels)["f1_macro"]
            row = dict(params)
            row["validation_macro_f1"] = score
            search_log.append(row)
            if score > best_score:
                best_score = score
                chosen = dict(params)
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
        class_weight=config.RF_DEFAULTS["class_weight"],
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
        "class_weight": config.RF_DEFAULTS["class_weight"],
        "bootstrap": True,
        "random_state": seed,
        "search_n_estimators": config.SEARCH_N_ESTIMATORS if search else None,
    }
    metrics = {
        "data_source": source,
        "warning": config.SYNTHETIC_WARNING if source != "REAL" else "",
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
        "train": train_stats,
        "test": test_stats,
        "baseline_accuracy": baseline,
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
    }
    model = FloodModel(forest, preprocess, feature_names, labels, list(X_train.columns), extra)
    return {
        "model": model,
        "metrics": _jsonable(metrics),
        "impurity_importance": ranked_importance(feature_names, forest.feature_importances_),
        "permutation_importance": perm,
        "search": search_log,
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
        f"TEST  accuracy={metrics['test']['accuracy']:.3f} "
        f"macro-F1={metrics['test']['f1_macro']:.3f} "
        f"precision={metrics['test']['precision_macro']:.3f} "
        f"recall={metrics['test']['recall_macro']:.3f}"
    )
    print(f"majority baseline accuracy={metrics['baseline_accuracy']:.3f}")
    print(f"OOB accuracy={metrics['oob_score']}")
    print(f"train-test accuracy gap={metrics['overfit_gap_accuracy']:.3f}")
    print(f"fit+search seconds={metrics['train_seconds']:.1f}")
    print(metrics["test_report"])
    print(config.DISCLAIMER)


def main() -> None:
    if not config.DATASET_CSV.exists():
        raise SystemExit("Missing data/processed/dataset.csv. Run: python scripts/build_dataset.py")
    frame = load_processed()
    result = train_dataframe(frame, search=True, explain=True, verbose=True)
    if config.DATASET_META.exists():
        meta = json.loads(config.DATASET_META.read_text(encoding="utf-8"))
        rule = meta.get("label_rule")
        if rule:
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
