"""Feature formulas. A column is added only when every input it needs is present.

Formulas
--------
rainfall_x_impervious
    rainfall * impervious_area / 100
    Rainfall depth (mm) times the impervious fraction. A runoff-depth proxy that
    ignores infiltration on the remaining pervious surface.

cumulative_rainfall
    rainfall + rainfall_antecedent
    Period rainfall plus the previous period. Skipped when antecedent rainfall
    was not supplied. A single rainfall column is not copied into a fake total.

elevation_relative_to_water
    elevation * distance_to_water
    The schema has no surveyed water-surface elevation, so this is not a height
    above water. It is the product of elevation (m) and distance (km). Small
    values mean low ground close to a water body. Units are mixed on purpose;
    the forest only uses the rank of the product.
"""

from __future__ import annotations

import pandas as pd

FORMULAS = {
    "rainfall_x_impervious": "rainfall * impervious_area / 100",
    "cumulative_rainfall": "rainfall + rainfall_antecedent",
    "elevation_relative_to_water": "elevation * distance_to_water",
}


class FeatureEngineer:
    """Stateless formulas. Nothing is fit, so there is no statistic to leak."""

    def transform(self, frame: pd.DataFrame) -> pd.DataFrame:
        out = frame.copy()
        if "rainfall" in out.columns and "impervious_area" in out.columns:
            out["rainfall_x_impervious"] = out["rainfall"] * out["impervious_area"] / 100.0
        if "rainfall" in out.columns and "rainfall_antecedent" in out.columns:
            out["cumulative_rainfall"] = out["rainfall"] + out["rainfall_antecedent"]
        if "elevation" in out.columns and "distance_to_water" in out.columns:
            out["elevation_relative_to_water"] = out["elevation"] * out["distance_to_water"]
        return out

    def engineered_columns(self, frame: pd.DataFrame) -> list[str]:
        created = []
        if "rainfall" in frame.columns and "impervious_area" in frame.columns:
            created.append("rainfall_x_impervious")
        if "rainfall" in frame.columns and "rainfall_antecedent" in frame.columns:
            created.append("cumulative_rainfall")
        if "elevation" in frame.columns and "distance_to_water" in frame.columns:
            created.append("elevation_relative_to_water")
        return created
