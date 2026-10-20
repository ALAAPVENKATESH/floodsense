"""Design tokens for FloodSense. Charts and CSS both read this module."""

from __future__ import annotations

from dataclasses import dataclass

BRAND = "#4F46E5"
BRAND_HOVER = "#4338CA"

RISK_COLORS = {
    "LOW": "#16A34A",
    "MODERATE": "#F59E0B",
    "HIGH": "#DC2626",
}

RISK_TINT = {
    "light": {"LOW": "#DCFCE7", "MODERATE": "#FEF3C7", "HIGH": "#FEE2E2"},
    "dark": {"LOW": "#052E16", "MODERATE": "#451A03", "HIGH": "#450A0A"},
}

UNITS = {
    "rainfall": "mm",
    "rainfall_intensity": "mm/h",
    "rainfall_antecedent": "mm",
    "elevation": "m",
    "slope": "°",
    "drainage_density": "km/km²",
    "impervious_area": "%",
    "distance_to_water": "km",
    "historical_flood": "0–1",
}

LABELS = {
    "rainfall": "Rainfall",
    "rainfall_intensity": "Rainfall intensity",
    "rainfall_antecedent": "Antecedent rainfall",
    "elevation": "Elevation",
    "slope": "Slope",
    "drainage_density": "Drainage density",
    "impervious_area": "Impervious area",
    "distance_to_water": "Distance to water",
    "historical_flood": "Historical flood",
    "rainfall_x_impervious": "Rainfall × impervious",
    "cumulative_rainfall": "Cumulative rainfall",
    "elevation_relative_to_water": "Elevation × distance",
}

GROUPS = (
    ("Rainfall", ("rainfall", "rainfall_intensity", "rainfall_antecedent")),
    ("Terrain", ("elevation", "slope", "distance_to_water")),
    ("Urban & Drainage", ("impervious_area", "drainage_density")),
    ("History", ("historical_flood",)),
)

STEPS = {
    "rainfall": 1.0,
    "rainfall_intensity": 0.1,
    "rainfall_antecedent": 1.0,
    "elevation": 1.0,
    "slope": 0.1,
    "drainage_density": 0.01,
    "impervious_area": 1.0,
    "distance_to_water": 0.01,
    "historical_flood": 0.01,
}

NAV = (
    ("dashboard", "Dashboard", "▦"),
    ("predict", "Predict", "◈"),
    ("batch", "Batch", "▤"),
    ("insights", "Insights", "◉"),
    ("dataset", "Dataset", "▥"),
    ("about", "About", "ⓘ"),
)

PIPELINE = (
    "Data collection",
    "Cleaning",
    "Feature engineering",
    "Decision trees",
    "Random forest",
    "Prediction",
    "Evaluation",
)


@dataclass(frozen=True)
class ChartTheme:
    """Colors Plotly needs so a chart matches the active mode."""

    mode: str
    text: str
    muted: str
    grid: str
    brand: str
    surface: str


TOKENS = {
    "light": {
        "bg": "#F8FAFC",
        "surface": "#FFFFFF",
        "surface2": "#F1F5F9",
        "sidebar": "#FFFFFF",
        "text": "#0F172A",
        "muted": "#64748B",
        "border": "#E2E8F0",
        "brand": BRAND,
        "brandHover": BRAND_HOVER,
        "shadow": "0 1px 2px rgba(15, 23, 42, 0.04), 0 12px 32px rgba(15, 23, 42, 0.06)",
        "shadowHover": "0 10px 28px rgba(15, 23, 42, 0.10)",
        "bannerBg": "#FFFBEB",
        "bannerBorder": "#FDE68A",
        "bannerText": "#92400E",
        "scroll": "#CBD5E1",
        "selection": "#C7D2FE",
        "lowBg": RISK_TINT["light"]["LOW"],
        "modBg": RISK_TINT["light"]["MODERATE"],
        "highBg": RISK_TINT["light"]["HIGH"],
    },
    "dark": {
        "bg": "#0B1020",
        "surface": "#12182B",
        "surface2": "#1A2238",
        "sidebar": "#0E1426",
        "text": "#E5E7EB",
        "muted": "#94A3B8",
        "border": "#243049",
        "brand": "#6366F1",
        "brandHover": "#818CF8",
        "shadow": "0 12px 32px rgba(0, 0, 0, 0.35)",
        "shadowHover": "0 16px 36px rgba(0, 0, 0, 0.45)",
        "bannerBg": "#3A2A08",
        "bannerBorder": "#92400E",
        "bannerText": "#FDE68A",
        "scroll": "#334155",
        "selection": "#3730A3",
        "lowBg": RISK_TINT["dark"]["LOW"],
        "modBg": RISK_TINT["dark"]["MODERATE"],
        "highBg": RISK_TINT["dark"]["HIGH"],
    },
}


def chart_theme(mode: str) -> ChartTheme:
    """Return the Plotly palette for ``light`` or ``dark``."""
    token = TOKENS["dark" if mode == "dark" else "light"]
    return ChartTheme(
        mode="dark" if mode == "dark" else "light",
        text=token["text"],
        muted=token["muted"],
        grid=token["border"],
        brand=token["brand"],
        surface=token["surface"],
    )


def display_name(feature: str) -> str:
    """Human label for a column name."""
    return LABELS.get(feature, feature.replace("_", " ").title())
