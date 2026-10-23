"""Write data/raw/flood_classification.csv from flood_dataset_classification.csv.

All 6,237 rows are kept (both flood-positive and non-flood events).

Labelling rule (risk_class written directly so train.py uses it as-is):
  - occured = 0                  →  LOW    (no recorded flood event)
  - occured = 1, Rainfall <= p33 →  MODERATE  (flood, lower rainfall tercile)
  - occured = 1, Rainfall >  p33 →  HIGH      (flood, upper two terciles)

The p33 cut is computed on the flood-positive rows of the FULL file.
Because this script is run before the train/test split, the cut is not
derived from training-only data.  AGENTS.md §Data says: "If a real file
already has risk_class, keep it."  Writing risk_class here means train.py
will use the column directly without fitting any threshold.

The feature set for this run: rainfall_file, elevation, slope, latitude,
longitude, year.  Total Deaths, duration, Total Affected, distance, and
Disaster Type are excluded as predictors.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import config

SOURCE = ROOT / "flood_dataset_classification.csv"
OUT = config.DATA_RAW / "flood_classification.csv"


def prepare(source: Path = SOURCE, out: Path = OUT) -> pd.DataFrame:
    frame = pd.read_csv(source)

    # ------------------------------------------------------------------ #
    # Compute p33 threshold on flood-positive rows only                   #
    # This is a single global constant derived from the full file, so it  #
    # does not adapt to the training split and cannot leak the test set.  #
    # ------------------------------------------------------------------ #
    flood_mask = frame["occured"] == 1
    p33 = float(frame.loc[flood_mask, "Rainfall"].quantile(0.33))

    # ------------------------------------------------------------------ #
    # Assign labels                                                        #
    # ------------------------------------------------------------------ #
    labels = pd.Series("LOW", index=frame.index)
    labels[flood_mask & (frame["Rainfall"] <= p33)] = "MODERATE"
    labels[flood_mask & (frame["Rainfall"] > p33)] = "HIGH"

    table = pd.DataFrame(
        {
            "row_id": frame.index.astype(int),
            "latitude": frame["Latitude"],
            "longitude": frame["Longitude"],
            "period": frame["time"].astype(int).astype(str),
            "year": frame["time"].astype(int),
            "rainfall_file": frame["Rainfall"],
            "elevation": frame["Elevation"],
            "slope": frame["Slope"],
            "risk_class": labels,
            "data_source": config.CLASSIFICATION_SOURCE,
        }
    )

    out.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(out, index=False)

    rain_hi = int((table["rainfall_file"] > config.FEATURE_RANGES["rainfall"][1]).sum())
    elev_lo = int((table["elevation"] < config.FEATURE_RANGES["elevation"][0]).sum())

    print(f"wrote {out.name}: {len(table)} rows from {len(frame)}")
    print(f"  p33 rainfall cut (flood rows): {p33:.1f} mm")
    print(f"  LOW:      {(labels == 'LOW').sum()}")
    print(f"  MODERATE: {(labels == 'MODERATE').sum()}")
    print(f"  HIGH:     {(labels == 'HIGH').sum()}")
    print(f"  rainfall above {config.FEATURE_RANGES['rainfall'][1]:.0f} mm: {rain_hi}")
    print(f"  elevation below {config.FEATURE_RANGES['elevation'][0]:.0f} m: {elev_lo}")
    return table


if __name__ == "__main__":
    prepare()
