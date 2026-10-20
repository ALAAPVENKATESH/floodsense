# Dataset documentation

Subject: Machine Learning (BIC703), Group G2.

The training table used for the numbers in the report is `flood_dataset_classification.csv`. It is not a surveyed flood record. The coursework simulator remains the fallback when `data/raw/` has no CSV.

## Current build

| Item | Value |
| --- | --- |
| File | `data/processed/dataset.csv` from `data/raw/flood_classification.csv` |
| Source | `flood_dataset_classification.csv` in the project root, prepared by `scripts/prepare_classification.py` |
| `data_source` | `CLASSIFICATION` on every row |
| What a row is | One flood-positive record (`occured = 1`, Disaster Type 0) |
| Rows in the source file | 6,237, of which 4,192 are flood-positive |
| Rows used | 4,186. Six duplicate rows dropped. 770 extreme measurements blanked, not deleted |
| Classes | LOW 1,738, MODERATE 1,401, HIGH 1,047, from the spatial training-split cuts |
| Columns kept | rainfall_file, elevation, slope, latitude, longitude, year, plus CHIRPS columns and population columns (all missing) |
| Columns not used | deaths, people affected (label only), disaster type, occured, duration, distance |
| `time` | Integer calendar year, 1900–2016. Not an epoch and not `YYYYMM`. No month or day |
| Geography | About 53°S–70°N and 172°W–178°E. 484 cells of 5° on the flood rows |
| Label | `Total Affected`. Spatial-training cuts: LOW ≤ 3,000 people, MODERATE ≤ 55,700, else HIGH. Quantiles 0.40 and 0.75 fit on the training split only |
| Features in the saved model | `rainfall_file`, `elevation`, `slope`, `latitude`, `longitude`, four CHIRPS columns, `rain_missing`, three population columns, `pop_missing`, `pop_year_clipped` |
| Features present but not in the saved model | `year`, `rainfall_missing`, `elevation_missing` |
| Features absent | rainfall intensity, antecedent rainfall, drainage density, impervious area, distance to water, historical flood. Not filled in |
| Licence | None is cited in the file |

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

## External rainfall and population

Joined in `scripts/build_external_features.py` on `row_id` (the source-file index of each flood-positive row), using latitude, longitude, and year. A pixel that cannot be read is NaN plus a missing indicator. Nothing is guessed.

Coverage is counted on the 4,192 flood-positive rows, before the six duplicate drops.

| Check | Rows | Share |
| --- | ---: | ---: |
| Year in 1981–2016 (CHIRPS exists) | 3,686 | 87.9% |
| That year and latitude inside 50°S–50°N | 3,510 | 83.7% |
| Year in 2000–2020 (WorldPop epoch exists) | 2,474 | 59.0% |

Decade counts on the flood rows: 1900: 3, 1910: 2, 1920: 9, 1930: 13, 1940: 24, 1950: 65, 1960: 136, 1970: 215, 1980: 425, 1990: 826, 2000: 1,557, 2010: 917.

| Source | Version | Resolution | Years | Licence | Access |
| --- | --- | --- | --- | --- | --- |
| CHIRPS | v2.0, Climate Hazards Center, UC Santa Barbara | 0.05°, 50°S–50°N | Monthly, 1981–present. This run uses 1981–2016. The anomaly baseline is 1981–2010 | Public domain. Cite [Funk et al. 2015](https://doi.org/10.1038/sdata.2015.66) | [Data page](https://www.chc.ucsb.edu/data/chirps). Files: [chirps-v2.0.YYYY.MM.tif.gz](https://data.chc.ucsb.edu/products/CHIRPS-2.0/global_monthly/tifs/) |
| WorldPop | Unconstrained global mosaics, 1 km aggregated. Not downloaded for this run | 30 arc-second, people per pixel | 2000–2020 yearly. Event years before 2000 would clip to 2000 and set `pop_year_clipped` | CC BY 4.0. [WorldPop and CIESIN (2018)](https://doi.org/10.5258/SOTON/WP00647) | [Listing](https://hub.worldpop.org/geodata/listing?id=64). Files: [ppp_YYYY_1km_Aggregated.tif](https://data.worldpop.org/GIS/Population/Global_2000_2020/) |
| GPW v4 (fallback only) | Population density, revision 11, NASA SEDAC | 30 arc-second | Epochs 2000, 2005, 2010, 2015, 2020 | CC BY 4.0. Free NASA Earthdata login | [GPW v4 Rev.11](https://sedac.ciesin.columbia.edu/data/set/gpw-v4-population-density-rev11) |
| ERA5 (fallback only) | Monthly averaged reanalysis, total precipitation | 0.25° | 1981–2016 would be requested | Copernicus licence. Free CDS account | [Climate Data Store](https://cds.climate.copernicus.eu/) |

CHIRPS annual rainfall is the mean of a 0.25° window (5 cells of 0.05°), because the file coordinates are often event or region centroids. `rain_anomaly` is `(annual − 1981–2010 mean) / sample standard deviation` at that same window. The baseline is built from CHIRPS alone. It does not use the label. When the flood year is inside 1981–2010, that year is inside the baseline.

Population density is people per pixel divided by the latitude-corrected cell area (`111.32 km` per degree). `pop_sum_25km` and `pop_sum_100km` sum pixel counts inside a circle in local kilometres. That is an approximation, not a surveyed buffer. A GPW file, if WorldPop is absent, is already a density and is multiplied by cell area before the sum.

Rasters stay in `data/external/` and are not committed. `data/external/MANIFEST.json` records the URL, retrieval date, size, and checksum. `data/processed/external_features.csv` is the small table that is merged.

The file's own rainfall, after values above 2,000 mm and values repeated on more than 1% of the flood rows are set aside, is compared with CHIRPS in `external_meta.json`. That comparison uses the full flood table. The model refits the filled-value list on the training split only. On 2,542 comparable rows the Pearson correlation is 0.799. WorldPop was not downloaded for this run, so every population measurement is missing and `pop_missing` is 1. `pop_year_clipped` only marks years outside 2000–2020 (41.0% of the flood rows). It is not a population measurement. `rain_missing` is 16.5%.
