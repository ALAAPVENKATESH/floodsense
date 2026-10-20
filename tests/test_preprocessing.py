"""Preprocessing, features, and the raw-CSV build path."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import config
from flood_rf.features import FeatureEngineer
from flood_rf.preprocessing import (
    Preprocessor,
    bin_score_train_quantiles,
    bin_unit_score,
    stratified_kfold_indices,
    stratified_train_test_split,
)
from build_dataset import build


def test_median_is_fit_on_train_only():
    train = pd.DataFrame({"rainfall": [1.0, 3.0, np.nan]})
    test = pd.DataFrame({"rainfall": [np.nan, 100.0]})
    pre = Preprocessor().fit(train)
    filled = pre.transform(test)
    assert pre.medians_["rainfall"] == pytest.approx(2.0)
    assert filled["rainfall"].iloc[0] == pytest.approx(2.0)
    leaked = Preprocessor().fit(pd.concat([train, test], ignore_index=True))
    assert leaked.medians_["rainfall"] != pytest.approx(pre.medians_["rainfall"])


def test_clip_bounds_come_from_train():
    train = pd.DataFrame({"a": [1.0, 2.0, 3.0, 4.0]})
    pre = Preprocessor(clip_outliers=True).fit(train)
    filled = pre.transform(pd.DataFrame({"a": [100.0]}))
    assert filled["a"].iloc[0] == pytest.approx(5.5)


def test_categorical_mode_and_unknown_code():
    train = pd.DataFrame({"zone": ["a", "b", "a", None]})
    pre = Preprocessor().fit(train)
    assert pre.modes_["zone"] == "a"
    filled = pre.transform(pd.DataFrame({"zone": [None, "b", "c"]}))
    assert filled["zone"].tolist() == [0.0, 1.0, -1.0]


def test_stratified_split_keeps_every_class():
    y = np.array(["LOW"] * 20 + ["MODERATE"] * 10 + ["HIGH"] * 10)
    X = pd.DataFrame({"rainfall": np.arange(len(y), dtype=float)})
    X_train, X_test, y_train, y_test = stratified_train_test_split(X, y, test_size=0.25, random_state=1)
    assert set(y_train) == set(y)
    assert set(y_test) == set(y)
    assert len(X_train) + len(X_test) == len(y)
    assert abs(len(X_test) / len(y) - 0.25) < 0.05


def test_kfold_covers_each_row_once():
    y = np.array([0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2])
    seen = []
    for train, val in stratified_kfold_indices(y, n_splits=4, random_state=0):
        assert len(np.intersect1d(train, val)) == 0
        seen.append(val)
    assert np.array_equal(np.sort(np.concatenate(seen)), np.arange(len(y)))


def test_train_quantiles_ignore_the_other_split():
    train = np.arange(10, dtype=float)
    other = np.array([100.0])
    _, other_labels, cuts = bin_score_train_quantiles(train, other, quantiles=(0.40, 0.75))
    assert cuts == pytest.approx(tuple(np.quantile(train, [0.40, 0.75])))
    assert other_labels[0] == "HIGH"
    joined = np.quantile(np.concatenate([train, other]), [0.40, 0.75])
    assert cuts[1] != pytest.approx(joined[1])


def test_unit_score_uses_fixed_cuts():
    labels = bin_unit_score(pd.Series([0.1, 0.33, 0.5, 0.66, 0.9]))
    assert labels.tolist() == ["LOW", "LOW", "MODERATE", "MODERATE", "HIGH"]


def test_engineered_formulas_and_guards():
    frame = pd.DataFrame(
        {
            "rainfall": [10.0],
            "impervious_area": [50.0],
            "rainfall_antecedent": [3.0],
            "elevation": [100.0],
            "distance_to_water": [4.0],
        }
    )
    out = FeatureEngineer().transform(frame)
    assert out["rainfall_x_impervious"].iloc[0] == pytest.approx(5.0)
    assert out["cumulative_rainfall"].iloc[0] == pytest.approx(13.0)
    assert out["elevation_relative_to_water"].iloc[0] == pytest.approx(400.0)
    thin = FeatureEngineer().transform(pd.DataFrame({"elevation": [1.0]}))
    assert "cumulative_rainfall" not in thin.columns
    assert "rainfall_x_impervious" not in thin.columns
    assert "elevation_relative_to_water" not in thin.columns


def test_empty_raw_folder_writes_labeled_synthetic(tmp_path, capsys):
    meta = build(raw_dir=tmp_path / "raw", out_dir=tmp_path / "out", n_synthetic=200, seed=1)
    captured = capsys.readouterr()
    assert "SYNTHETIC" in captured.err
    assert meta["data_source"] == "SYNTHETIC"
    frame = pd.read_csv(tmp_path / "out" / "dataset.csv")
    assert set(frame["data_source"]) == {"SYNTHETIC"}
    assert (tmp_path / "out" / "flood_synthetic.csv").exists()
    assert set(frame["risk_class"]) <= set(config.CLASS_LABELS)


def test_real_csv_keeps_supplied_label_and_normalizes_names(tmp_path, capsys):
    raw = tmp_path / "raw"
    raw.mkdir()
    pd.DataFrame(
        {
            "Rainfall mm": [10, 80, 200, 20, 300, 40],
            "Elevation": [400, 80, 10, 300, 5, 120],
            "Impervious Area": [10, 40, 90, 15, 85, 30],
            "Historical Flood": [0.1, 0.2, 0.4, 0.05, 0.5, 0.2],
            "risk": ["LOW", "medium", "HIGH", "LOW", "HIGH", "MODERATE"],
        }
    ).to_csv(raw / "stations.csv", index=False)
    meta = build(raw_dir=raw, out_dir=tmp_path / "out")
    frame = pd.read_csv(tmp_path / "out" / "dataset.csv")
    assert meta["data_source"] == "REAL"
    assert "SYNTHETIC" not in capsys.readouterr().err
    assert set(frame["data_source"]) == {"REAL"}
    assert "rainfall" in frame.columns
    assert "historical_flood" in frame.columns
    assert set(frame["risk_class"]) <= set(config.CLASS_LABELS)
    assert (frame["risk_class"] == "MODERATE").any()


def test_unit_hazard_column_is_removed_from_predictors(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    pd.DataFrame(
        {
            "rainfall": [10, 50, 100, 200],
            "historical_flood": [0.1, 0.4, 0.7, 0.9],
        }
    ).to_csv(raw / "hazard.csv", index=False)
    meta = build(raw_dir=raw, out_dir=tmp_path / "out")
    frame = pd.read_csv(tmp_path / "out" / "dataset.csv")
    assert "historical_flood" not in frame.columns
    assert frame["risk_class"].tolist() == ["LOW", "MODERATE", "HIGH", "HIGH"]
    assert meta["excluded_features"] == ["historical_flood"]


def test_invalid_physical_value_is_dropped(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    pd.DataFrame(
        {
            "rainfall": [10, -5, 30],
            "risk_class": ["LOW", "LOW", "HIGH"],
        }
    ).to_csv(raw / "bad.csv", index=False)
    meta = build(raw_dir=raw, out_dir=tmp_path / "out")
    frame = pd.read_csv(tmp_path / "out" / "dataset.csv")
    assert meta["dropped_invalid"] == 1
    assert len(frame) == 2
    assert Path(tmp_path / "out" / "dataset_meta.json").exists()
