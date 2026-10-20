"""Sample CHIRPS and WorldPop (or GPW) onto flood rows. No fabricated values.

Writes data/processed/external_features.csv keyed by row_id. A pixel that
cannot be read stays NaN and the matching missing indicator is 1.

CHIRPS annual rainfall is the mean of a 0.25° window, because the file
coordinates are often event or region centroids. The anomaly is the annual
window total minus the 1981–2010 mean at that window, divided by that
period's standard deviation. The event year is inside the baseline when the
flood year is 1981–2010. That baseline uses only CHIRPS, not the label.

Population uses the nearest file in 2000–2020. Years outside that range clip
to the edge and set pop_year_clipped = 1.
"""

from __future__ import annotations

import argparse
import json
import sys
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
from external_sample import (
    CHIRPS_WINDOW_DEG,
    POP_YEAR_MAX,
    POP_YEAR_MIN,
    annual_from_months,
    half_pixels,
    open_array,
    population_at_point,
    population_year,
    row_col,
    window_mean,
    window_means,
)
from flood_rf.preprocessing import filled_rainfall_values

SOURCE = ROOT / "flood_dataset_classification.csv"
CHIRPS_DIR = ROOT / "data" / "external" / "chirps"
WORLDPOP_DIR = ROOT / "data" / "external" / "worldpop"
GPW_DIR = ROOT / "data" / "external" / "gpw"
OUT = config.DATA_PROCESSED / "external_features.csv"
META = config.DATA_PROCESSED / "external_meta.json"


def load_flood(source: Path = SOURCE) -> pd.DataFrame:
    frame = pd.read_csv(source)
    flood = frame.loc[frame["occured"] == 1].copy()
    return pd.DataFrame(
        {
            "row_id": flood.index.astype(int),
            "latitude": pd.to_numeric(flood["Latitude"], errors="coerce"),
            "longitude": pd.to_numeric(flood["Longitude"], errors="coerce"),
            "year": flood["time"].astype(int),
            "rainfall_file": pd.to_numeric(flood["Rainfall"], errors="coerce"),
        }
    ).reset_index(drop=True)


def _chirps_months(year: int) -> list[Path]:
    return [CHIRPS_DIR / f"chirps-v2.0.{year}.{month:02d}.tif.gz" for month in range(1, 13)]


def _population_files() -> tuple[dict[int, Path], bool]:
    """Return {year: path} and whether the raster stores people per pixel."""
    found: dict[int, Path] = {}
    if WORLDPOP_DIR.exists():
        for path in sorted(WORLDPOP_DIR.glob("ppp_*_1km_Aggregated.tif")):
            found[int(path.name.split("_")[1])] = path
    if found:
        return found, True
    if GPW_DIR.exists():
        import re

        for path in sorted(GPW_DIR.glob("*.tif")):
            years = re.findall(r"(?:19|20)\d{2}", path.name)
            if years:
                found[int(years[-1])] = path
    return found, False


def _locate(grid: dict, lat: np.ndarray, lon: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    height, width = grid["array"].shape
    rows = np.full(len(lat), -1, dtype=int)
    cols = np.full(len(lat), -1, dtype=int)
    for i, (y, x) in enumerate(zip(lat, lon)):
        if not np.isfinite(y) or not np.isfinite(x):
            continue
        loc = row_col(grid["west"], grid["north"], grid["xsize"], grid["ysize"], width, height, x, y)
        if loc is not None:
            rows[i], cols[i] = loc
    return rows, cols


def _sample_chirps(frame: pd.DataFrame) -> dict[str, np.ndarray]:
    n = len(frame)
    annual = np.full(n, np.nan)
    wet = np.full(n, np.nan)
    share = np.full(n, np.nan)
    anomaly = np.full(n, np.nan)
    clim_sum = np.zeros(n)
    clim_sq = np.zeros(n)
    clim_n = np.zeros(n)
    lat = frame["latitude"].to_numpy(dtype=float)
    lon = frame["longitude"].to_numpy(dtype=float)
    year = frame["year"].to_numpy(dtype=int)
    eligible = (year >= 1981) & (year <= 2016) & (lat >= -50) & (lat <= 50)
    rows = cols = None
    half = None
    for event_year in range(1981, 2017):
        paths = _chirps_months(event_year)
        if not all(path.exists() for path in paths):
            missing = sum(not path.exists() for path in paths)
            print(f"CHIRPS {event_year}: {missing} month file(s) missing, year left as NaN", flush=True)
            continue
        bands = []
        grid = None
        for path in paths:
            grid = open_array(path)
            bands.append(np.asarray(grid["array"], dtype=np.float32))
        if rows is None:
            rows, cols = _locate(grid, lat, lon)
            half = half_pixels(CHIRPS_WINDOW_DEG, grid["xsize"])
            print(f"CHIRPS window {CHIRPS_WINDOW_DEG}° is {2 * half + 1} pixels on a side", flush=True)
        nodata = grid["nodata"]
        full = grid["full_globe"]
        monthly = np.column_stack(
            [
                window_means(bands[month], rows, cols, half, nodata, full, minimum=0)
                for month in range(12)
            ]
        )
        complete = np.isfinite(monthly).all(axis=1)
        totals = np.full(n, np.nan)
        wet_year = np.full(n, np.nan)
        share_year = np.full(n, np.nan)
        totals[complete] = monthly[complete].sum(axis=1)
        wet_year[complete] = monthly[complete].max(axis=1)
        positive = complete & (totals > 0)
        share_year[positive] = wet_year[positive] / totals[positive]
        if 1981 <= event_year <= 2010:
            base = np.isfinite(totals)
            clim_sum[base] += totals[base]
            clim_sq[base] += totals[base] ** 2
            clim_n[base] += 1
        event = eligible & (year == event_year)
        annual[event] = totals[event]
        wet[event] = wet_year[event]
        share[event] = share_year[event]
        finite = int(np.isfinite(totals[eligible]).sum())
        print(f"CHIRPS {event_year}: {finite} eligible points sampled", flush=True)
        del bands
    std = np.full(n, np.nan)
    mean = np.full(n, np.nan)
    enough = clim_n >= 2
    mean[enough] = clim_sum[enough] / clim_n[enough]
    var = np.zeros(n)
    var[enough] = (clim_sq[enough] - (clim_sum[enough] ** 2) / clim_n[enough]) / (clim_n[enough] - 1)
    std[enough & (var > 0)] = np.sqrt(var[enough & (var > 0)])
    usable = np.isfinite(annual) & np.isfinite(std) & (std > 0)
    anomaly[usable] = (annual[usable] - mean[usable]) / std[usable]
    missing = (~np.isfinite(annual)).astype(float)
    return {
        "rain_annual_mm": annual,
        "rain_max_month_mm": wet,
        "rain_wet_month_share": share,
        "rain_anomaly": anomaly,
        "rain_missing": missing,
    }


def _sample_population(frame: pd.DataFrame) -> dict[str, np.ndarray]:
    n = len(frame)
    density = np.full(n, np.nan)
    sum_25 = np.full(n, np.nan)
    sum_100 = np.full(n, np.nan)
    missing = np.ones(n, dtype=float)
    source_year = np.full(n, np.nan)
    files, people_per_pixel = _population_files()
    years = frame["year"].to_numpy(dtype=int)
    clipped = np.array([population_year(int(y), files)[1] for y in years], dtype=float)
    if not files:
        print("no WorldPop or GPW rasters; population left as NaN", flush=True)
        return {
            "pop_density_point": density,
            "pop_sum_25km": sum_25,
            "pop_sum_100km": sum_100,
            "pop_missing": missing,
            "pop_year_clipped": clipped,
            "pop_source_year": source_year,
        }
    assigned = []
    for raw in years:
        nearest, _flag = population_year(int(raw), files)
        assigned.append(-1 if nearest is None else nearest)
    assigned = np.asarray(assigned, dtype=int)
    lat = frame["latitude"].to_numpy(dtype=float)
    lon = frame["longitude"].to_numpy(dtype=float)
    label = "WorldPop people/pixel" if people_per_pixel else "GPW density"
    for epoch in sorted(set(assigned.tolist()) - {-1}):
        grid = open_array(files[epoch])
        rows, cols = _locate(grid, lat, lon)
        use = np.flatnonzero(assigned == epoch)
        for i in use:
            if rows[i] < 0:
                continue
            dens, s25, s100, flag = population_at_point(
                grid["array"],
                int(rows[i]),
                int(cols[i]),
                float(lat[i]),
                float(grid["ysize"]),
                grid["nodata"],
                grid["full_globe"],
                people_per_pixel,
            )
            density[i] = dens
            sum_25[i] = s25
            sum_100[i] = s100
            missing[i] = flag
            source_year[i] = epoch
        print(f"{label} {epoch}: {int((assigned == epoch).sum())} rows", flush=True)
        del grid
    return {
        "pop_density_point": density,
        "pop_sum_25km": sum_25,
        "pop_sum_100km": sum_100,
        "pop_missing": missing,
        "pop_year_clipped": clipped,
        "pop_source_year": source_year,
    }


def _correlation(file_rain: pd.Series, chirps: np.ndarray) -> dict:
    """Diagnostic only. The model still detects filled rainfall on the training split."""
    filled = filled_rainfall_values(file_rain)
    rain = pd.to_numeric(file_rain, errors="coerce")
    rain = rain.mask(rain > config.FEATURE_RANGES["rainfall"][1])
    if filled:
        rain = rain.mask(rain.isin(filled))
    chirps = pd.Series(chirps, index=rain.index)
    both = rain.notna() & np.isfinite(chirps.to_numpy())
    n = int(both.sum())
    pearson = None
    if n >= 3:
        pearson = float(np.corrcoef(rain[both], chirps[both])[0, 1])
    weak = pearson is None or abs(pearson) < 0.3
    return {
        "n": n,
        "pearson": pearson,
        "weak": bool(weak),
        "filled_values_full_table": filled,
        "note": (
            "Filled values here are detected on all flood rows for this comparison only. "
            "The model refits that list on the training split."
        ),
    }


def _decade_table(frame: pd.DataFrame, features: pd.DataFrame) -> list[dict]:
    decade = (frame["year"] // 10 * 10).astype(int)
    rows = []
    for value, idx in decade.groupby(decade).groups.items():
        block = features.loc[list(idx)]
        rows.append(
            {
                "decade": int(value),
                "n": int(len(block)),
                "rain_missing": float(block["rain_missing"].mean()),
                "pop_missing": float(block["pop_missing"].mean()),
                "pop_year_clipped": float(block["pop_year_clipped"].mean()),
            }
        )
    return rows


def lookup_point(lat: float, lon: float, year: int) -> dict:
    """One location for the Predict page. Missing rasters stay null."""
    out = {
        "rain_annual_mm": None,
        "rain_max_month_mm": None,
        "rain_wet_month_share": None,
        "pop_density_point": None,
        "pop_sum_25km": None,
        "pop_sum_100km": None,
        "pop_year_clipped": int(year < POP_YEAR_MIN or year > POP_YEAR_MAX),
        "rain_missing": 1,
        "pop_missing": 1,
        "note": "",
    }
    paths = _chirps_months(int(year)) if 1981 <= int(year) <= 2016 else []
    if paths and all(path.exists() for path in paths):
        bands = [open_array(path) for path in paths]
        grid = bands[0]
        loc = row_col(
            grid["west"], grid["north"], grid["xsize"], grid["ysize"],
            grid["array"].shape[1], grid["array"].shape[0], lon, lat,
        )
        if loc is not None and -50 <= lat <= 50:
            half = half_pixels(CHIRPS_WINDOW_DEG, grid["xsize"])
            monthly = [
                window_mean(band["array"], loc[0], loc[1], half, band["nodata"], band["full_globe"], minimum=0)
                for band in bands
            ]
            total, wet, share = annual_from_months(monthly)
            if np.isfinite(total):
                out["rain_annual_mm"] = total
                out["rain_max_month_mm"] = wet
                out["rain_wet_month_share"] = share
                out["rain_missing"] = 0
        else:
            out["note"] = "Location is outside the CHIRPS grid."
    elif int(year) < 1981:
        out["note"] = "CHIRPS starts in 1981."
    files, people_per_pixel = _population_files()
    epoch, clipped = population_year(int(year), files)
    out["pop_year_clipped"] = clipped
    if epoch is not None:
        grid = open_array(files[epoch])
        loc = row_col(
            grid["west"], grid["north"], grid["xsize"], grid["ysize"],
            grid["array"].shape[1], grid["array"].shape[0], lon, lat,
        )
        if loc is not None:
            dens, s25, s100, flag = population_at_point(
                grid["array"], loc[0], loc[1], lat, grid["ysize"], grid["nodata"], grid["full_globe"], people_per_pixel
            )
            out["pop_missing"] = flag
            if flag == 0:
                out["pop_density_point"] = dens
                out["pop_sum_25km"] = s25
                out["pop_sum_100km"] = s100
    cleaned = {}
    for key, value in out.items():
        if isinstance(value, (float, np.floating)):
            value = float(value)
            cleaned[key] = None if not np.isfinite(value) else value
        elif isinstance(value, (int, np.integer)):
            cleaned[key] = int(value)
        else:
            cleaned[key] = value
    return cleaned


def build(source: Path = SOURCE, out: Path = OUT) -> pd.DataFrame:
    frame = load_flood(source)
    rain = _sample_chirps(frame)
    pop = _sample_population(frame)
    table = pd.DataFrame({"row_id": frame["row_id"].to_numpy()})
    for column, values in {**rain, **pop}.items():
        table[column] = values
    if list(table["row_id"]) != list(frame["row_id"]) or len(table) != len(frame):
        raise RuntimeError("external feature table changed the flood row order")
    out.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(out, index=False)
    corr = _correlation(frame["rainfall_file"], table["rain_annual_mm"].to_numpy())
    decades = _decade_table(frame, table)
    meta = {
        "n_rows": int(len(table)),
        "chirps": {
            "dataset": "CHIRPS v2.0",
            "resolution": "0.05 degree",
            "window_deg": CHIRPS_WINDOW_DEG,
            "climatology": "1981-2010 mean and standard deviation at the same 0.25° window",
            "anomaly": "(annual - mean) / std, sample standard deviation",
        },
        "population": {
            "primary": "WorldPop unconstrained global 1 km mosaic, people per pixel",
            "fallback": "GPW v4 population density if data/external/gpw has GeoTIFFs and WorldPop does not",
            "radius": "25 km and 100 km sums use a local kilometre circle, 111.32 km per degree",
            "year_min": POP_YEAR_MIN,
            "year_max": POP_YEAR_MAX,
        },
        "missing_share": {column: float(table[column].mean()) for column in ("rain_missing", "pop_missing", "pop_year_clipped")},
        "decades": decades,
        "file_vs_chirps": corr,
    }
    META.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"wrote {out} ({len(table)} rows)", flush=True)
    print("missing share", meta["missing_share"], flush=True)
    print("file rainfall vs CHIRPS", corr, flush=True)
    if corr["weak"]:
        print("FLAG: correlation with the file rainfall is weak. Do not treat the file rainfall as CHIRPS.", flush=True)
    print(pd.DataFrame(decades).to_string(index=False), flush=True)
    return table


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lookup", nargs=3, metavar=("LAT", "LON", "YEAR"))
    args = parser.parse_args()
    if args.lookup:
        lat, lon, year = (float(args.lookup[0]), float(args.lookup[1]), int(float(args.lookup[2])))
        json.dump(lookup_point(lat, lon, year), sys.stdout)
        print()
        return
    if not SOURCE.exists():
        raise SystemExit(f"Missing {SOURCE}")
    build()


if __name__ == "__main__":
    main()
