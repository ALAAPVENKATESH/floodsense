"""External rainfall and population features. No network."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from build_dataset import merge_external
from external_eval import _paired, feature_sets, run_sets
from external_sample import population_year, sample_raster_file, window_mean, window_means
from fetch_external import download_file
from flood_rf.data_loader import model_feature_columns
from flood_rf.preprocessing import Preprocessor, apply_measurement_mask, split_positions
from train import train_dataframe

_BANNED = {"total_affected", "deaths", "disaster_type", "occured", "occurred", "flood_occurrence"}


def test_geotiff_point_bounds_and_nodata(tmp_path):
    import rasterio
    from rasterio.transform import from_origin

    values = np.full((6, 6), -9999, dtype=np.float32)
    values[1, 2] = 42
    path = tmp_path / "tiny.tif"
    with rasterio.open(
        path, "w", driver="GTiff", height=6, width=6, count=1, dtype="float32",
        crs="EPSG:4326", transform=from_origin(-3, 3, 1, 1), nodata=-9999,
    ) as dst:
        dst.write(values, 1)
    value, missing = sample_raster_file(path, lon=-0.5, lat=1.5)
    assert value == pytest.approx(42)
    assert missing == 0
    value, missing = sample_raster_file(path, lon=20, lat=1.5)
    assert np.isnan(value) and missing == 1
    value, missing = sample_raster_file(path, lon=-2.5, lat=2.5)
    assert np.isnan(value) and missing == 1


def test_year_clip_sets_the_flag():
    year, clipped = population_year(1990, [2000, 2010])
    assert year == 2000 and clipped == 1
    year, clipped = population_year(2010, [2000, 2010])
    assert year == 2010 and clipped == 0
    masked = apply_measurement_mask(
        pd.DataFrame({"year": [1990, 2010], "pop_density_point": [1.0, 2.0]}),
        [],
    )
    assert masked["pop_year_clipped"].tolist() == [1.0, 0.0]


def test_longitude_wrap_and_window_mean():
    wrapped = np.zeros((3, 8))
    wrapped[1, 0] = 10
    wrapped[1, 7] = 30
    assert window_mean(wrapped, 1, 0, half=1, full_globe=True) == pytest.approx(40 / 9)
    assert window_mean(wrapped, 1, 0, half=1, full_globe=False) == pytest.approx(10 / 6)
    block = np.zeros((5, 5))
    block[1:4, 1:4] = 1
    block[2, 2] = 0
    assert window_mean(block, 2, 2, half=1) == pytest.approx(8 / 9)


def test_many_windows_match_the_scalar_mean():
    rng = np.random.RandomState(0)
    values = rng.random((8, 10))
    rows = np.array([1, 3, 7, -1])
    cols = np.array([0, 9, 4, 1])
    many = window_means(values, rows, cols, half=1, full_globe=True)
    for index, (row, col) in enumerate(zip(rows[:3], cols[:3])):
        assert many[index] == pytest.approx(window_mean(values, int(row), int(col), half=1, full_globe=True))
    assert np.isnan(many[3])


def test_merge_keeps_count_and_order():
    frame = pd.DataFrame({"row_id": [3, 1, 2], "elevation": [1, 2, 3]})
    external = pd.DataFrame({"row_id": [2, 3, 1], "rain_annual_mm": [9.0, 8.0, 7.0]})
    merged = merge_external(frame, external)
    assert merged["row_id"].tolist() == [3, 1, 2]
    assert merged["rain_annual_mm"].tolist() == [8.0, 7.0, 9.0]
    assert len(merged) == 3


def test_imputer_median_comes_from_train_only():
    train = pd.DataFrame({"rain_annual_mm": [1.0, 3.0, np.nan]})
    held = pd.DataFrame({"rain_annual_mm": [np.nan, 1000.0]})
    filled = Preprocessor().fit(train).transform(held)
    assert filled.loc[0, "rain_annual_mm"] == pytest.approx(2.0)
    assert filled.loc[1, "rain_annual_mm"] == pytest.approx(1000.0)


def test_feature_list_never_includes_the_label():
    frame = pd.DataFrame(
        {
            "rainfall_file": [1.0],
            "rain_annual_mm": [2.0],
            "pop_density_point": [3.0],
            "total_affected": [4.0],
            "deaths": [5.0],
            "disaster_type": [0],
            "occured": [1],
            "flood_hazard_score": [6.0],
        }
    )
    assert set(model_feature_columns(frame)).isdisjoint(_BANNED | {"flood_hazard_score"})


def _fixture(n: int = 80) -> pd.DataFrame:
    rng = np.random.RandomState(0)
    return pd.DataFrame(
        {
            "rainfall_file": rng.uniform(10, 80, n),
            "elevation": rng.uniform(5, 100, n),
            "slope": rng.uniform(0, 10, n),
            "latitude": np.repeat([10.0, 16.0, 22.0, 28.0], n // 4),
            "longitude": np.repeat([70.0, 76.0, 82.0, 88.0], n // 4),
            "year": rng.randint(2000, 2016, n),
            "rain_annual_mm": rng.uniform(100, 800, n),
            "rain_max_month_mm": rng.uniform(20, 200, n),
            "rain_wet_month_share": rng.uniform(0.1, 0.4, n),
            "rain_anomaly": rng.normal(0, 1, n),
            "pop_density_point": rng.uniform(10, 500, n),
            "pop_sum_25km": rng.uniform(1e3, 1e5, n),
            "pop_sum_100km": rng.uniform(1e4, 1e6, n),
            "rain_missing": np.zeros(n),
            "pop_missing": np.zeros(n),
            "pop_year_clipped": np.zeros(n),
            "flood_hazard_score": rng.uniform(1, 1000, n),
            "total_affected": rng.uniform(1, 1000, n),
            "deaths": rng.uniform(0, 10, n),
            "disaster_type": np.zeros(n, dtype=int),
            "occured": np.ones(n, dtype=int),
            "data_source": ["CLASSIFICATION"] * n,
        }
    )


def test_ablation_uses_identical_cells(monkeypatch):
    calls = {"n": 0}
    real = split_positions

    def wrapped(*args, **kwargs):
        calls["n"] += 1
        return real(*args, **kwargs)

    monkeypatch.setattr("train.split_positions", wrapped)
    frame = _fixture()
    rows = run_sets(
        frame,
        feature_sets(frame.columns),
        {"max_depth": 3, "min_samples_leaf": 2, "max_features": "sqrt", "class_weight": None},
        1,
        [42, 7],
        "A_all",
    )
    assert calls["n"] == 2
    for seed in (42, 7):
        positions = [row["positions"][0].tolist() for row in rows if row["seed"] == seed]
        assert positions and all(item == positions[0] for item in positions)
        assert {row["feature_set"] for row in rows if row["seed"] == seed} >= {
            "physical",
            "plus_lat_lon",
            "plus_chirps",
            "plus_population",
            "full",
            "no_lat_lon",
        }


def test_tiny_fixture_trains_without_the_label():
    result = train_dataframe(
        _fixture(),
        search=False,
        explain=False,
        n_estimators_final=2,
        split="spatial",
        fixed_params={"max_depth": 3, "min_samples_leaf": 2, "max_features": "sqrt", "class_weight": None},
    )
    names = set(result["metrics"]["features"])
    assert names.isdisjoint(_BANNED | {"flood_hazard_score"})
    assert "rain_annual_mm" in names
    assert "pop_density_point" in names


def test_paired_comparison_is_empty_without_rows():
    assert _paired([], "B_covered") == {"per_seed": []}


def test_download_is_mocked(tmp_path, monkeypatch):
    calls = {"get": 0}

    class _Response:
        status_code = 200
        headers = {"Content-Length": "1000"}

        def raise_for_status(self):
            return None

        def iter_content(self, chunk_size=1024):
            yield b"a" * 1000

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

    def fake_head(url, **kwargs):
        return _Response()

    def fake_get(url, **kwargs):
        calls["get"] += 1
        return _Response()

    monkeypatch.setattr("fetch_external.requests.head", fake_head)
    monkeypatch.setattr("fetch_external.requests.get", fake_get)
    dest = tmp_path / "chirps-v2.0.2010.01.tif.gz"
    first = download_file("http://example.test/chirps", dest, retries=1)
    second = download_file("http://example.test/chirps", dest, retries=1)
    assert first["status"] == "downloaded"
    assert dest.read_bytes() == b"a" * 1000
    assert second["status"] == "skipped"
    assert calls["get"] == 1
