# Dataset documentation

Subject: Machine Learning (BIC703), Group G2.

The training table used for the numbers in the report is **synthetic**. Replace this section's "current build" notes when measured files are added, and keep the source, version, coverage, period, and licence next to the file.

## Current build

| Item | Value |
| --- | --- |
| File | `data/processed/dataset.csv` and `data/processed/flood_synthetic.csv` |
| `data_source` | `SYNTHETIC` on every row |
| Rows | 8,000 |
| Generator seed | 42 |
| Classes | LOW 2,747, MODERATE 2,655, HIGH 2,598 |
| Measured files used | none (`data/raw/` had no CSV) |
| Licence of the simulator output | coursework use inside this project; not a public dataset |

`data/processed/dataset_meta.json` repeats the source flag and the label rule.

## What a real extract should record

Fill these in when a CSV is added. Do not leave them blank in the final report if the numbers come from measured data.

| Source | Version / layer | Geographic coverage | Time period | Licence | Columns taken |
| --- | --- | --- | --- | --- | --- |
| National Water Data Portal / India-WRIS (rainfall) |  |  |  |  |  |
| ISRO / NRSC Bhuvan (elevation, slope, drainage, land cover, flood layers) |  |  |  |  |  |
| Other |  |  |  |  |  |

The loader does not download these portals. `data/raw/README.md` describes the column names and the merge key (`location_id`, or `district` + `period`, or latitude/longitude + `period`).

## Label rule for a measured file

1. If `risk_class` is already in the file, it is the label. `medium` and similar words are mapped to `MODERATE`.
2. Else if `flood_hazard_score` or `historical_flood` lies on 0–1, fixed cuts are used: `<= 0.33` LOW, `<= 0.66` MODERATE, otherwise HIGH. Those cuts do not depend on the sample. The column that was binned is removed from the predictors so the label is not a copy of one input.
3. Else the score is kept and `scripts/train.py` bins it at the 0.40 and 0.75 quantiles **of the training split only**, applies those same cuts to the test split, and drops the score column.

Rows outside `config.FEATURE_RANGES` are dropped. Missing measurements stay missing until training, where medians (numeric) or modes (text) are fit on the training split.

## Synthetic generator

Implemented in `scripts/generate_synthetic_data.py`. The docstring there is the full rule. Short version:

Features are random draws with fixed families (gamma elevation, lognormal rainfall, and so on), then rounded. A latent score is a weighted sum of scaled drivers plus `Normal(0, 0.04)` noise. Weights:

| Driver | Weight | Direction |
| --- | ---: | --- |
| rainfall (scaled 0–400 mm) | 0.24 | higher raises risk |
| low elevation (scaled 0–500 m, then inverted) | 0.20 | lower ground raises risk |
| impervious area | 0.14 | higher raises risk |
| short distance to water (0–20 km, inverted) | 0.10 | closer raises risk |
| rainfall intensity (0–80 mm/h) | 0.08 | higher raises risk |
| antecedent rainfall (0–300 mm) | 0.08 | higher raises risk |
| low drainage density (0.2–4.5, inverted) | 0.08 | sparser drainage raises risk |
| historical flood indicator | 0.08 | higher raises risk |

Slope is not inside the score. It is generated as `0.035 * elevation` plus exponential noise, so it only tracks elevation.

Labels, before noise, are bins of that score on the generated table: at or below the 0.34 quantile is LOW, above the 0.67 quantile is HIGH, and the rest is MODERATE. Then 6% of labels are cycled LOW to MODERATE to HIGH and back to LOW. Then 4% of the cells in `rainfall_intensity`, `drainage_density`, `distance_to_water`, and `rainfall_antecedent` are set to missing.

The quantile cuts are part of the simulator. They are not statistics fit inside `train.py`. The forest never sees the latent score.

Do not report accuracy on this table as performance on a real city.
