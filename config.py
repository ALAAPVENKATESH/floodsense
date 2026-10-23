"""Seeds, paths, feature schema, and model defaults for the flood-risk project."""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parent
RANDOM_STATE = 42

DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"
DATASET_CSV = DATA_PROCESSED / "dataset.csv"
SYNTHETIC_CSV = DATA_PROCESSED / "flood_synthetic.csv"
DATASET_META = DATA_PROCESSED / "dataset_meta.json"

MODELS_DIR = ROOT / "models"
MODEL_PATH = MODELS_DIR / "model.json"
METRICS_PATH = MODELS_DIR / "metrics.json"
IMPORTANCE_PATH = MODELS_DIR / "feature_importance.json"

FIGURES_DIR = ROOT / "reports" / "figures"

CLASS_LABELS = ["LOW", "MODERATE", "HIGH"]
TARGET = "risk_class"
SOURCE_COLUMN = "data_source"
BENCHMARK_SOURCE = "URBANFLOODBENCH"
# Peak depth cuts for UrbanFloodBench. Fixed, not fit on the sample.
BENCHMARK_DEPTH_M = (0.15, 0.50)
BENCHMARK_MAX_ROWS = 8000

# Predictors the model may use. Anything absent from a real CSV is dropped.
# NOTE: rainfall_file is EXCLUDED -- it is the exact column used to assign
# risk_class labels (p33 threshold on flood rows). Including it is direct label leakage.
BASE_FEATURES = [
    "rainfall",
    "rainfall_intensity",
    "elevation",
    "slope",
    "drainage_density",
    "impervious_area",
    "distance_to_water",
    "historical_flood",
]
# CHIRPS rainfall and WorldPop/GPW exposure. Absent columns are omitted.
RAIN_EXT = [
    "rain_annual_mm",
    "rain_max_month_mm",
    "rain_wet_month_share",
    "rain_anomaly",
]
POP_EXT = [
    "pop_density_point",
    "pop_sum_25km",
    "pop_sum_100km",
]
# NOTE: rain_missing and pop_missing are EXCLUDED -- rain_missing=1 for every
# LOW row (occured=0 has no CHIRPS data), so it perfectly encodes the target.
# pop_year_clipped is also excluded as a zero-information constant.
EXTERNAL_FLAGS: list[str] = []  # formerly ["rain_missing", "pop_missing", "pop_year_clipped"]
OPTIONAL_FEATURES = [
    "rainfall_antecedent",
    "latitude",
    "longitude",
    "year",
    "rainfall_missing",
    "elevation_missing",
    *RAIN_EXT,
    *POP_EXT,
    *EXTERNAL_FLAGS,
]
INDICATOR_FEATURES = ["rainfall_missing", "elevation_missing"]

# Not model inputs. Used only to build a location key or a label.
# latitude, longitude, and year are also optional predictors when the file has them.
ID_COLUMNS = ["location_id", "district", "period", "latitude", "longitude", "row_id", "pop_source_year"]
LABEL_SCORE_COLUMNS = ["flood_hazard_score"]

# Fixed bins for a 0-1 hazard score. Not fit on the sample, so the test
# set cannot move the cut points. See docs/dataset_documentation.md.
UNIT_SCORE_THRESHOLDS = (0.33, 0.66)

# If the score is not on 0-1, train.py bins it with these train-only quantiles.
TRAIN_SCORE_QUANTILES = (0.40, 0.75)

# Spatial block for the headline split. 5 degrees is about 550 km.
GRID_CELL_DEG = 5.0
GROUP_FOLDS = 4
# A rainfall value seen in more than this share of the training rows is treated as filled.
FILLED_VALUE_MIN_SHARE = 0.01
VARIANCE_SEEDS = (42, 7, 11, 99, 123)

COLUMN_ALIASES = {
    "rain": "rainfall",
    "rainfall_mm": "rainfall",
    "precip": "rainfall",
    "precipitation": "rainfall",
    "precipitation_mm": "rainfall",
    "intensity": "rainfall_intensity",
    "rain_intensity": "rainfall_intensity",
    "rainfall_intensity_mm_hr": "rainfall_intensity",
    "elev": "elevation",
    "elevation_m": "elevation",
    "dem": "elevation",
    "slope_deg": "slope",
    "terrain_slope": "slope",
    "drain_density": "drainage_density",
    "drainage": "drainage_density",
    "built_up": "impervious_area",
    "builtup": "impervious_area",
    "built_up_percent": "impervious_area",
    "impervious": "impervious_area",
    "impervious_percent": "impervious_area",
    "dist_water": "distance_to_water",
    "distance_water": "distance_to_water",
    "distance_to_river": "distance_to_water",
    "flood_history": "historical_flood",
    "flood_hazard": "historical_flood",
    "hazard": "flood_hazard_score",
    "hazard_score": "flood_hazard_score",
    "flood_hazard_score": "flood_hazard_score",
    "antecedent_rainfall": "rainfall_antecedent",
    "antecedent": "rainfall_antecedent",
    "label": "risk_class",
    "flood_risk": "risk_class",
    "risk": "risk_class",
    "lat": "latitude",
    "lon": "longitude",
    "lng": "longitude",
}

LABEL_SYNONYMS = {
    "LOW": "LOW",
    "L": "LOW",
    "0": "LOW",
    "MODERATE": "MODERATE",
    "MEDIUM": "MODERATE",
    "MED": "MODERATE",
    "MOD": "MODERATE",
    "1": "MODERATE",
    "HIGH": "HIGH",
    "H": "HIGH",
    "2": "HIGH",
}

# Physical sanity bounds. Rows outside these are dropped at build time.
# They are not statistical estimates and are not learned from the sample.
FEATURE_RANGES = {
    "rainfall": (0.0, 2000.0),
    "rainfall_intensity": (0.0, 300.0),
    "rainfall_antecedent": (0.0, 2000.0),
    "elevation": (-20.0, 9000.0),
    "slope": (0.0, 90.0),
    "drainage_density": (0.0, 20.0),
    "impervious_area": (0.0, 100.0),
    "distance_to_water": (0.0, 500.0),
    "historical_flood": (0.0, 1.0),
    "flood_hazard_score": (0.0, 1.0e6),
    "latitude": (-90.0, 90.0),
    "longitude": (-180.0, 180.0),
    "year": (1800.0, 2100.0),
    "rainfall_missing": (0.0, 1.0),
    "elevation_missing": (0.0, 1.0),
    "rainfall_file": (0.0, 2000.0),
    "rain_annual_mm": (0.0, 20000.0),
    "rain_max_month_mm": (0.0, 5000.0),
    "rain_wet_month_share": (0.0, 1.0),
    "rain_anomaly": (-100.0, 100.0),
    "pop_density_point": (0.0, 1.0e6),
    "pop_sum_25km": (0.0, 1.0e9),
    "pop_sum_100km": (0.0, 1.0e10),
    "rain_missing": (0.0, 1.0),
    "pop_missing": (0.0, 1.0),
    "pop_year_clipped": (0.0, 1.0),
}

FEATURE_HELP = {
    "rainfall": "Rainfall amount over the selected period (mm).",
    "rainfall_intensity": "Rainfall rate (mm/h).",
    "rainfall_antecedent": "Rainfall in the preceding period (mm).",
    "elevation": "Ground elevation (m).",
    "slope": "Terrain slope (degrees).",
    "drainage_density": "Drainage density (km per km²).",
    "impervious_area": "Built-up or impervious surface (percent).",
    "distance_to_water": "Distance to the relevant water body (km).",
    "historical_flood": "Historical flood indicator, scaled 0 to 1.",
    "latitude": "Latitude (degrees). Taken from the source row, not geocoded.",
    "longitude": "Longitude (degrees). Taken from the source row, not geocoded.",
    "year": "Calendar year parsed from the integer year column.",
    "rainfall_missing": "1 when rainfall was blank, above 2,000 mm, or a value repeated on more than 1% of the training rows.",
    "elevation_missing": "1 when elevation was blank or below -20 m.",
    "rainfall_file": "Rainfall from the classification file (mm), after the filled-value and 2,000 mm mask.",
    "rain_annual_mm": "CHIRPS v2.0 annual total (mm), mean of a 0.25° window. Not event rainfall; only the year is known.",
    "rain_max_month_mm": "CHIRPS v2.0 wettest month in that year (mm), same 0.25° window.",
    "rain_wet_month_share": "Wettest CHIRPS month divided by the annual total.",
    "rain_anomaly": "CHIRPS annual total minus the 1981–2010 mean at that cell, divided by that period's standard deviation.",
    "pop_density_point": "WorldPop people per km² at the point (1 km pixel count divided by latitude-corrected cell area). GPW density is used only if WorldPop is absent.",
    "pop_sum_25km": "WorldPop people within 25 km. Approximate circle in local kilometres, not a surveyed buffer.",
    "pop_sum_100km": "WorldPop people within 100 km, same approximation as the 25 km sum.",
    "rain_missing": "1 when CHIRPS has no value (year before 1981, outside 50°S–50°N, or nodata).",
    "pop_missing": "1 when the population raster has no value.",
    "pop_year_clipped": "1 when the event year is before 2000 or after 2020 and the population year was clipped to that range.",
}

MERGE_KEYS = ["location_id", "district"]

RF_DEFAULTS = {
    "n_estimators": 40,
    "max_depth": 8,
    "min_samples_split": 2,
    "min_samples_leaf": 8,
    "max_features": "sqrt",
    "criterion": "gini",
    "bootstrap": True,
    "class_weight": "balanced",
    "oob_score": True,
    "n_jobs": 1,
    "random_state": RANDOM_STATE,
}

# Structural search uses fewer trees; the chosen params are refit at n_estimators.
SEARCH_N_ESTIMATORS = 15
# Structure only. n_estimators and class_weight are compared afterwards on the
# winning row, using the same training folds. A full cartesian product of this
# forest is too slow to finish inside a coursework run.
SEARCH_GRID = [
    {"max_depth": 6, "min_samples_leaf": 12, "max_features": "sqrt"},
    {"max_depth": 8, "min_samples_leaf": 8, "max_features": "sqrt"},
    {"max_depth": 10, "min_samples_leaf": 5, "max_features": "sqrt"},
    {"max_depth": 8, "min_samples_leaf": 8, "max_features": "log2"},
    {"max_depth": 8, "min_samples_leaf": 8, "max_features": 0.5},
    {"max_depth": 8, "min_samples_leaf": 8, "max_features": 1.0},
]
SEARCH_N_ESTIMATOR_CHOICES = (20, 40)
SEARCH_CLASS_WEIGHTS = (None, "balanced")
VALIDATION_FRACTION = 0.2
TEST_SIZE = 0.2

SYNTHETIC_WARNING = (
    "SYNTHETIC DATA: this dataset was simulated for coursework. "
    "It is not measured rainfall, terrain, or flood history. "
    "Do not report these metrics as real-world performance."
)

BENCHMARK_WARNING = (
    "URBANFLOODBENCH: these rows are HEC-RAS simulations from "
    "Kaggle jiayulim/urbanfloodbench, not surveyed flood observations. "
    "Do not report these metrics as real-world performance."
)

BENCHMARK_LABEL_RULE = (
    "Peak simulated depth is water-surface elevation minus cell elevation, "
    "converted from feet to metres. LOW if depth < 0.15 m, MODERATE if depth < 0.50 m, "
    "otherwise HIGH. The cuts are fixed. Depth is not a model input."
)

BENCHMARK_COVERAGE = (
    "Kaggle jiayulim/urbanfloodbench, competition models 1 and 2. "
    "HEC-RAS rain-on-grid simulations. US customary units converted to metres and millimetres. "
    "Rainfall events are model forcings, not a gauge record."
)

CLASSIFICATION_SOURCE = "CLASSIFICATION"
CLASSIFICATION_WARNING = (
    "CLASSIFICATION: flood_dataset_classification.csv, flood-positive rows only. "
    "Classes are train-only quantiles of people affected, not a surveyed water depth. "
    "Rainfall contains repeated filled values. "
    "Do not report these metrics as real-world performance."
)
CLASSIFICATION_LABEL_RULE = (
    "Flood-positive rows only (occured = 1, which matches Disaster Type 0). "
    "Total Affected is the score. scripts/train.py bins it with quantiles "
    f"{TRAIN_SCORE_QUANTILES} fit on the training split only, then drops it. "
    "Deaths, affected, disaster type, and occured are not model inputs."
)
CLASSIFICATION_COVERAGE = (
    "Local file flood_dataset_classification.csv. Global coordinates and a year column. "
    "No agency or licence is cited in the file. "
    "Distance was left out: most values are 0 and the rest are far outside 0–500 km."
)


def warning_for(source: str) -> str:
    if source == "SYNTHETIC":
        return SYNTHETIC_WARNING
    if source == BENCHMARK_SOURCE:
        return BENCHMARK_WARNING
    if source == CLASSIFICATION_SOURCE:
        return CLASSIFICATION_WARNING
    return ""

DISCLAIMER = (
    "Academic prediction system for Machine Learning (BIC703), Group G2. "
    "This is not an official emergency-warning or evacuation system. "
    "Do not use it for operational decisions, alerts, or evacuation."
)

CAUSATION_NOTE = (
    "Feature importance reflects model behavior on the fitted data, "
    "not proof of causation."
)
