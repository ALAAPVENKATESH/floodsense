"""Write data/raw/flood_classification.csv from flood_dataset_classification.csv.

Flood-positive rows only (occured = 1, Disaster Type 0). Total Affected is the
label score. Deaths, disaster type, occured, duration, and distance are not
copied. Distance is mostly zero and the rest sit far outside 0–500 km.
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
    flood = frame.loc[frame["occured"] == 1].copy()
    table = pd.DataFrame(
        {
            "row_id": flood.index.astype(int),
            "latitude": flood["Latitude"],
            "longitude": flood["Longitude"],
            "period": flood["time"].astype(int).astype(str),
            "year": flood["time"].astype(int),
            "rainfall_file": flood["Rainfall"],
            "elevation": flood["Elevation"],
            "slope": flood["Slope"],
            "flood_hazard_score": flood["Total Affected"],
            "data_source": config.CLASSIFICATION_SOURCE,
        }
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(out, index=False)
    rain_hi = int((table["rainfall_file"] > config.FEATURE_RANGES["rainfall"][1]).sum())
    elev_lo = int((table["elevation"] < config.FEATURE_RANGES["elevation"][0]).sum())
    print(f"wrote {out.name}: {len(table)} flood rows from {len(frame)}")
    print(f"rainfall above {config.FEATURE_RANGES['rainfall'][1]:.0f} mm: {rain_hi}")
    print(f"elevation below {config.FEATURE_RANGES['elevation'][0]:.0f} m: {elev_lo}")
    return table


if __name__ == "__main__":
    prepare()
