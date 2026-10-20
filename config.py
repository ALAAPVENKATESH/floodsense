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

# Predictors the model may use. Anything absent from a real CSV is dropped.
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
OPTIONAL_FEATURES = ["rainfall_antecedent"]

# Not model inputs. Used only to build a location key or a label.
ID_COLUMNS = ["location_id", "district", "period", "latitude", "longitude"]
LABEL_SCORE_COLUMNS = ["flood_hazard_score"]

# Fixed bins for a 0-1 hazard score. Not fit on the sample, so the test
# set cannot move the cut points. See docs/dataset_documentation.md.
UNIT_SCORE_THRESHOLDS = (0.33, 0.66)

# If the score is not on 0-1, train.py bins it with these train-only quantiles.
TRAIN_SCORE_QUANTILES = (0.40, 0.75)

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
SEARCH_GRID = [
    {"max_depth": 6, "min_samples_leaf": 12, "max_features": "sqrt"},
    {"max_depth": 8, "min_samples_leaf": 8, "max_features": "sqrt"},
    {"max_depth": 10, "min_samples_leaf": 5, "max_features": "sqrt"},
    {"max_depth": 8, "min_samples_leaf": 8, "max_features": 0.5},
]
VALIDATION_FRACTION = 0.2
TEST_SIZE = 0.2

SYNTHETIC_WARNING = (
    "SYNTHETIC DATA: this dataset was simulated for coursework. "
    "It is not measured rainfall, terrain, or flood history. "
    "Do not report these metrics as real-world performance."
)

DISCLAIMER = (
    "Academic prediction system for Machine Learning (BIC703), Group G2. "
    "This is not an official emergency-warning or evacuation system. "
    "Do not use it for operational decisions, alerts, or evacuation."
)

CAUSATION_NOTE = (
    "Feature importance reflects model behavior on the fitted data, "
    "not proof of causation."
)
