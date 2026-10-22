"""Sanity check only. The saved project model is never the sklearn forest.

Run after the dataset exists. Prints holdout accuracy for the project forest
and for sklearn.ensemble.RandomForestClassifier on the same split.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import config
from flood_rf.data_loader import load_processed, model_feature_columns
from flood_rf.features import FeatureEngineer
from flood_rf.metrics import accuracy
from flood_rf.preprocessing import Preprocessor, encode_labels, stratified_train_test_split
from flood_rf.random_forest import RandomForestClassifier
from sklearn.ensemble import RandomForestClassifier as SklearnForest


def main() -> None:
    print("BENCHMARK ONLY. sklearn is not the project model and is not saved.")
    frame = load_processed().head(2000)
    score_col = None
    if config.TARGET not in frame.columns:
        for candidate in ("flood_hazard_score", "historical_flood"):
            if candidate in frame.columns:
                score_col = candidate
                break
    columns = model_feature_columns(frame, drop=[score_col] if score_col else None)
    if score_col:
        from flood_rf.preprocessing import bin_score_train_quantiles, split_positions
        train_pos, test_pos = split_positions(frame, "random", config.RANDOM_STATE)
        train_df = frame.iloc[train_pos].reset_index(drop=True)
        test_df = frame.iloc[test_pos].reset_index(drop=True)
        y_train, y_test, _ = bin_score_train_quantiles(train_df[score_col], test_df[score_col])
        X_train, X_test = train_df[columns], test_df[columns]
    else:
        X_train, X_test, y_train, y_test = stratified_train_test_split(
            frame[columns], frame[config.TARGET], test_size=0.2, random_state=config.RANDOM_STATE
        )
    labels = [label for label in config.CLASS_LABELS if label in set(y_train) | set(y_test)]
    pre = Preprocessor().fit(X_train)
    engineer = FeatureEngineer()
    train_x = engineer.transform(pre.transform(X_train))
    test_x = engineer.transform(pre.transform(X_test))
    names = list(train_x.columns)
    y_train_enc, _ = encode_labels(list(y_train), labels)
    y_test_enc, _ = encode_labels(list(y_test), labels)
    ours = RandomForestClassifier(
        n_estimators=30,
        max_depth=12,
        min_samples_leaf=4,
        max_features="sqrt",
        class_weight="balanced",
        oob_score=False,
        random_state=config.RANDOM_STATE,
    )
    ours.fit(train_x[names].to_numpy(dtype=np.float64), y_train_enc, n_classes=len(labels))
    theirs = SklearnForest(
        n_estimators=30,
        max_depth=12,
        min_samples_leaf=4,
        max_features="sqrt",
        class_weight="balanced",
        random_state=config.RANDOM_STATE,
    )
    theirs.fit(train_x[names].to_numpy(dtype=np.float64), y_train_enc)
    ours_acc = accuracy(y_test_enc, ours.predict(test_x[names].to_numpy(dtype=np.float64)))
    theirs_acc = accuracy(y_test_enc, theirs.predict(test_x[names].to_numpy(dtype=np.float64)))
    print(f"rows used={len(frame)} features={names}")
    print(f"project forest holdout accuracy={ours_acc:.3f}")
    print(f"sklearn forest holdout accuracy={theirs_acc:.3f}")
    print("The project model remains the NumPy forest in src/flood_rf.")


if __name__ == "__main__":
    main()
