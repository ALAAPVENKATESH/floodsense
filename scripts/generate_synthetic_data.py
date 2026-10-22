"""Simulated urban flood-risk rows for the coursework demo.

This file does not contain measured rainfall, terrain, or flood history.
Every row is tagged data_source = SYNTHETIC. The rules below are the whole
generator. Class cuts are quantiles of the latent score on that table.

Latent score, after clipping each driver with the fixed bounds in `_unit`.
Weights sum to 1. Slope is omitted here because it is tied to elevation.

    score =
        0.24 * rainfall_unit
      + 0.08 * intensity_unit
      + 0.08 * antecedent_unit
      + 0.20 * (1 - elevation_unit)
      + 0.14 * impervious_unit
      + 0.10 * (1 - distance_unit)
      + 0.08 * (1 - drainage_unit)
      + 0.08 * historical_flood
      + Normal(0, 0.04)

    rainfall_unit uses bounds 0 and 400 mm, intensity 0 and 80 mm/h,
    antecedent 0 and 300 mm, elevation 0 and 500 m,
    drainage 0.2 and 4.5, impervious 0 and 100, distance 0 and 20 km.

Labels, before noise, are bins of that score on the generated table:

    below the 0.34 quantile -> LOW
    below the 0.67 quantile -> MODERATE
    otherwise HIGH

The cut points are part of the simulator. They are not learned inside training.
Then 6% of labels are cycled LOW -> MODERATE -> HIGH -> LOW.
Then 4% of the cells in rainfall_intensity, drainage_density,
distance_to_water, and rainfall_antecedent are set to missing.

Feature draws (seeded, then rounded, then scored):

    elevation ~ Gamma(2, 40), clipped to [2, 600] m
    slope ~ 0.035 * elevation + Exponential(2), clipped to [0, 35] degrees
    rainfall ~ LogNormal(4.2, 0.70), clipped to [0, 450] mm
    rainfall_antecedent ~ LogNormal(3.8, 0.70), clipped to [0, 350] mm
    rainfall_intensity ~ 0.06 * rainfall + LogNormal(1.8, 0.50), clipped to [0, 120]
    drainage_density ~ Uniform(0.2, 4.5)
    impervious_area ~ Normal(58, 18) + 12 * (1 - elevation_unit), clipped to [5, 98]
    distance_to_water ~ LogNormal(0.7, 0.85), clipped to [0.05, 25] km
    historical_flood ~ 0.10 + 0.40 * (1 - elevation_unit)
                       + 0.25 * (1 - distance_unit) + Normal(0, 0.10),
                       clipped to [0, 1]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import config

LABEL_QUANTILES = (0.34, 0.67)
LABEL_NOISE = 0.01
SCORE_NOISE = 0.02
MISSING_RATE = 0.04
N_DEFAULT = 8000

LABEL_RULE = (
    "Synthetic latent score binned at quantiles "
    f"{LABEL_QUANTILES[0]} and {LABEL_QUANTILES[1]} of the generated table "
    f"(LOW / MODERATE / HIGH), then {LABEL_NOISE:.0%} cycled label noise. "
    "Not a measured hazard class."
)


def _unit(values: np.ndarray, low: float, high: float) -> np.ndarray:
    return np.clip((values - low) / (high - low), 0.0, 1.0)


def generate(n: int = N_DEFAULT, seed: int = config.RANDOM_STATE) -> pd.DataFrame:
    """Return `n` simulated rows. The frame is labeled SYNTHETIC on every row."""
    if n < 30:
        raise ValueError("n must be at least 30 so each class can appear")
    rng = np.random.RandomState(seed)
    elevation = np.clip(rng.gamma(shape=2.0, scale=40.0, size=n), 2.0, 600.0)
    slope = np.clip(0.035 * elevation + rng.exponential(2.0, size=n), 0.0, 35.0)
    rainfall = np.clip(rng.lognormal(mean=4.2, sigma=0.70, size=n), 0.0, 450.0)
    antecedent = np.clip(rng.lognormal(mean=3.8, sigma=0.70, size=n), 0.0, 350.0)
    intensity = np.clip(0.06 * rainfall + rng.lognormal(mean=1.8, sigma=0.50, size=n), 0.0, 120.0)
    drainage = rng.uniform(0.2, 4.5, size=n)
    impervious = np.clip(rng.normal(58.0, 18.0, size=n) + 12.0 * (1.0 - _unit(elevation, 0.0, 500.0)), 5.0, 98.0)
    distance = np.clip(rng.lognormal(mean=0.7, sigma=0.85, size=n), 0.05, 25.0)
    historical = np.clip(
        0.10
        + 0.40 * (1.0 - _unit(elevation, 0.0, 500.0))
        + 0.25 * (1.0 - _unit(distance, 0.0, 20.0))
        + rng.normal(0.0, 0.10, size=n),
        0.0,
        1.0,
    )

    elevation = np.round(elevation, 2)
    slope = np.round(slope, 3)
    rainfall = np.round(rainfall, 2)
    antecedent = np.round(antecedent, 2)
    intensity = np.round(intensity, 3)
    drainage = np.round(drainage, 3)
    impervious = np.round(impervious, 2)
    distance = np.round(distance, 3)
    historical = np.round(historical, 4)

    # Weights sum to 1. Slope is not in the score; it is generated from elevation
    # so it remains a weaker stand-in for terrain.
    score = (
        0.24 * _unit(rainfall, 0.0, 400.0)
        + 0.08 * _unit(intensity, 0.0, 80.0)
        + 0.08 * _unit(antecedent, 0.0, 300.0)
        + 0.20 * (1.0 - _unit(elevation, 0.0, 500.0))
        + 0.14 * _unit(impervious, 0.0, 100.0)
        + 0.10 * (1.0 - _unit(distance, 0.0, 20.0))
        + 0.08 * (1.0 - _unit(drainage, 0.2, 4.5))
        + 0.08 * historical
        + rng.normal(0.0, SCORE_NOISE, size=n)
    )
    low_cut, high_cut = np.quantile(score, LABEL_QUANTILES)
    labels = np.full(n, "MODERATE", dtype=object)
    labels[score <= low_cut] = "LOW"
    labels[score > high_cut] = "HIGH"
    order = np.array(["LOW", "MODERATE", "HIGH"])
    index = {name: i for i, name in enumerate(order)}
    for row in np.flatnonzero(rng.rand(n) < LABEL_NOISE):
        labels[row] = order[(index[labels[row]] + 1) % 3]

    frame = pd.DataFrame(
        {
            "rainfall": rainfall,
            "rainfall_intensity": intensity,
            "elevation": elevation,
            "slope": slope,
            "drainage_density": drainage,
            "impervious_area": impervious,
            "distance_to_water": distance,
            "historical_flood": historical,
            "rainfall_antecedent": antecedent,
            "risk_class": labels,
            "data_source": "SYNTHETIC",
        }
    )
    for column in ("rainfall_intensity", "drainage_density", "distance_to_water", "rainfall_antecedent"):
        frame.loc[rng.rand(n) < MISSING_RATE, column] = np.nan
    return frame


def main() -> None:
    parser = argparse.ArgumentParser(description="Write simulated flood-risk rows. Not real measurements.")
    parser.add_argument("--n", type=int, default=N_DEFAULT)
    parser.add_argument("--seed", type=int, default=config.RANDOM_STATE)
    parser.add_argument("--out", type=Path, default=config.SYNTHETIC_CSV)
    args = parser.parse_args()
    print(config.SYNTHETIC_WARNING, file=sys.stderr)
    frame = generate(args.n, args.seed)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.out, index=False)
    print(frame["risk_class"].value_counts().to_string())
    print(f"Wrote {args.out} ({len(frame)} rows)")


if __name__ == "__main__":
    main()
