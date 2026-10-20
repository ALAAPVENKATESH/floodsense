"""Spatial blocks, filled-rainfall detection, and the label-column guard."""

import numpy as np
import pandas as pd

from flood_rf.preprocessing import (
    apply_measurement_mask,
    filled_rainfall_values,
    grid_cell_ids,
    group_kfold,
    group_train_test_split,
)
from train import train_dataframe


def test_grid_cell_ids_are_deterministic():
    first = grid_cell_ids([10.0, -10.0], [20.0, -20.0], cell_deg=5.0)
    second = grid_cell_ids([10.0, -10.0], [20.0, -20.0], cell_deg=5.0)
    assert list(first) == list(second) == ["2:4", "-2:-4"]


def test_group_split_does_not_share_a_cell():
    groups = np.array(["a"] * 6 + ["b"] * 6 + ["c"] * 6 + ["d"] * 6)
    train_idx, test_idx = group_train_test_split(groups, 0.25, 42, y=["LOW", "HIGH"] * 12)
    assert set(groups[train_idx]).isdisjoint(set(groups[test_idx]))
    assert len(train_idx) and len(test_idx)


def test_group_kfold_folds_are_disjoint():
    groups = np.array(["a"] * 4 + ["b"] * 4 + ["c"] * 4 + ["d"] * 4)
    folds = group_kfold(groups, 4, 3, y=["LOW", "MODERATE", "HIGH", "LOW"] * 4)
    seen = []
    for train_idx, val_idx in folds:
        assert set(groups[train_idx]).isdisjoint(set(groups[val_idx]))
        seen.append(set(groups[val_idx]))
    for i, left in enumerate(seen):
        for right in seen[i + 1 :]:
            assert left.isdisjoint(right)


def test_filled_values_are_fit_on_the_training_values_only():
    training = [1.0] * 10 + [2.0, 3.0]
    held_out = [9.0] * 50
    found = filled_rainfall_values(training, min_share=0.2)
    assert found == [1.0]
    assert 9.0 not in filled_rainfall_values(training, min_share=0.2)
    assert 9.0 in filled_rainfall_values(list(training) + held_out, min_share=0.2)


def test_indicators_mark_filled_high_rainfall_and_low_elevation():
    frame = pd.DataFrame(
        {"rainfall": [1.0, 2.0, 5000.0], "elevation": [10.0, -40.0, 3.0]}
    )
    masked = apply_measurement_mask(frame, [1.0])
    assert masked["rainfall_missing"].tolist() == [1.0, 0.0, 1.0]
    assert masked["elevation_missing"].tolist() == [0.0, 1.0, 0.0]
    assert pd.isna(masked.loc[0, "rainfall"])
    assert pd.isna(masked.loc[2, "rainfall"])
    assert pd.isna(masked.loc[1, "elevation"])
    assert masked.loc[1, "rainfall"] == 2.0


def test_pipeline_does_not_train_on_the_label_columns():
    rng = np.random.RandomState(0)
    n = 80
    frame = pd.DataFrame(
        {
            "rainfall": rng.uniform(10, 80, n),
            "elevation": rng.uniform(5, 100, n),
            "slope": rng.uniform(0, 10, n),
            "latitude": np.repeat([10.0, 16.0, 22.0, 28.0], n // 4),
            "longitude": np.repeat([70.0, 76.0, 82.0, 88.0], n // 4),
            "year": rng.randint(1990, 2010, n),
            "flood_hazard_score": rng.uniform(1, 1000, n),
            "total_affected": rng.uniform(1, 1000, n),
            "deaths": rng.uniform(0, 10, n),
            "disaster_type": np.zeros(n, dtype=int),
            "occured": np.ones(n, dtype=int),
            "data_source": ["CLASSIFICATION"] * n,
        }
    )
    frame.loc[:15, "rainfall"] = 12.5
    result = train_dataframe(frame, search=False, explain=False, n_estimators_final=3, split="spatial")
    names = set(result["metrics"]["features"])
    assert names.isdisjoint({"total_affected", "deaths", "disaster_type", "occured", "flood_hazard_score"})
    assert "rainfall_missing" in names
    assert "latitude" in names
    assert "year" in names
