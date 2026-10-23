# FloodSense

Urban flood-risk prediction for one location: **LOW**, **MODERATE**, or **HIGH**.

Coursework for Machine Learning (BIC703), semester VII, group G2. The decision tree and the random forest are written in this repository with NumPy. scikit-learn is used only in `tests/` and in `scripts/benchmark_vs_sklearn.py`. It is not the model that gets saved or shown in the app.

Repository: [github.com/ALAAPVENKATESH/floodsense](https://github.com/ALAAPVENKATESH/floodsense)

**This is an academic prediction system. It is not an official emergency-warning or evacuation tool.**

## Data in this folder

The active saved model is trained on `flood_dataset_classification.csv` (the real historical disaster table in the project root), processed by `scripts/prepare_classification.py`. All 6,237 rows are kept. Labels are assigned using a fixed physical rule:

- `occured = 0` → **LOW** risk (no flood event recorded)
- `occured = 1` and `Rainfall ≤ 887.8 mm` → **MODERATE** risk (flood, lower rainfall tercile)
- `occured = 1` and `Rainfall > 887.8 mm` → **HIGH** risk (flood, upper two terciles)

The 887.8 mm cut is the 33rd percentile of the flood-positive rows' rainfall — a fixed global constant, not fit on the training split.

| Dataset | Used in the saved model | Description |
| --- | --- | --- |
| `flood_dataset_classification.csv` | **Yes (Active)** | 6,237 real historical disaster events (EM-DAT / CHIRPS joined). `data_source = CLASSIFICATION` |
| [CHIRPS v2.0](https://www.chc.ucsb.edu/data/chirps) | **Yes (joined)** | Annual and monthly satellite rainfall joined to each event by lat/lon/year |
| [WorldPop](https://hub.worldpop.org/geodata/listing?id=64) | Yes (joined) | 1 km population exposure mosaics joined to each event |
| Physical Urban Generator | Fallback only | Synthetic simulator used if `data/raw/` is empty |

- Processed table: `data/processed/dataset.csv` (6,034 rows after deduplication, 15 model features)
- Every row has `data_source = CLASSIFICATION`
- The Streamlit app displays an educational disclaimer. Do not report the scores as real-world operational guidance.
- To regenerate the dataset: run `python scripts/prepare_classification.py` then `python scripts/build_dataset.py`.

## Setup

Python 3.10 or newer. This copy was trained with Python 3.12.

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python scripts/prepare_classification.py
python scripts/fetch_external.py
python scripts/build_external_features.py
python scripts/build_dataset.py
python scripts/train.py
python -m pytest -q
streamlit run app/streamlit_app.py
```

`rasterio`, `requests`, and `tqdm` are used only by the download and sampling scripts. The forest does not import them. The rasterio wheel bundles GDAL, so a normal `pip install` is enough on Windows. If that wheel is unavailable, install rasterio from conda-forge (`conda install -c conda-forge rasterio`) and keep the other packages in this venv. Rasters are cached under `data/external/` and are not part of the git tree. If a host blocks a download, the fetch script prints the URL and the filename to place there by hand, then the sampler writes NaN plus a missing flag for anything it cannot read.

The headline search, spatial fit, random-split comparison, ablation, and five seeds took about 13 minutes on the machine used for this run.

## Data

Put measured CSVs in `data/raw/`. Column names, merge keys, and the label rule are in `data/raw/README.md` and `docs/dataset_documentation.md`.

Portals that are not in this run, if a later file is measured:

- Rainfall: [India-WRIS / National Water Data Portal](https://indiawris.gov.in/)
- Geographic and flood layers: [ISRO / NRSC Bhuvan](https://bhuvan.nrsc.gov.in/)

If `data/raw/` contains no CSV, `build_dataset.py` generates the synthetic table and says so. Do not copy that file back into `data/raw/`.

Columns the model can use, when they exist: `rainfall`, `rainfall_intensity`, `elevation`, `slope`, `drainage_density`, `impervious_area`, `distance_to_water`, `historical_flood`, and optional `rainfall_antecedent`. Missing optional columns are dropped. They are not invented.

Engineered columns, only when their inputs exist:

| Column | Formula |
| --- | --- |
| `rainfall_x_impervious` | `rainfall * impervious_area / 100` |
| `cumulative_rainfall` | `rainfall + rainfall_antecedent` |
| `elevation_relative_to_water` | `elevation * distance_to_water` |

`elevation_relative_to_water` is a product of metres and kilometres, not a surveyed height above water. The schema has no water-surface elevation.

## Pipeline

```mermaid
flowchart LR
    A[Data collection] --> B[Cleaning]
    B --> C[Feature engineering]
    C --> D[Decision trees]
    D --> E[Random forest]
    E --> F[Prediction]
    E --> G[Evaluation and figures]
```

When the table already has `risk_class`, a random split is stratified. For this run a spatial (5° grid cell) split separates the held-out test set. The medians and scalers are fit on the training side only. Hyperparameters are chosen by validation macro-F1 on training folds. The held-out test set is scored once for the chosen model.

## Results on the test holdout

Source: `CLASSIFICATION` (real historical disaster records from `flood_dataset_classification.csv`). Rows: 6,034 (train 4,820, test 1,214). Seed: 42. Classes: LOW 2,000 (train) / 413 (test), MODERATE 1,087 (train) / 251 (test), HIGH 2,146 (train) / 550 (test).

Labels: `occured = 0` → LOW. Flood events split at the 33rd rainfall percentile (887.8 mm) of flood rows → MODERATE / HIGH. Features include rainfall, elevation, slope, latitude, longitude, CHIRPS annual/monthly rainfall metrics, and WorldPop population exposure.

Chosen model: 20 trees, `max_depth=8`, `min_samples_leaf=8`, `max_features=1.0`, Gini, no class weight override. Out-of-bag accuracy: **0.997**. Majority-class baseline accuracy on the test set: **0.453**.

| Comparison | Accuracy | Macro F1 | QWK |
| --- | ---: | ---: | ---: |
| Test holdout (spatial split) | 0.996 | 0.995 | 0.997 |
| Training set performance | 0.997 | 0.997 | 0.998 |
| Majority class baseline | 0.453 | 0.208 | 0.000 |

Adjacent accuracy on the test set is **1.000** (100%). Severe-error rate (LOW called HIGH or HIGH called LOW) is **0.000** (0%).

Test confusion matrix (rows are true, columns are predicted LOW / MODERATE / HIGH):

|  | Pred LOW | Pred MODERATE | Pred HIGH |
| --- | ---: | ---: | ---: |
| True LOW | 413 | 0 | 0 |
| True MODERATE | 0 | 246 | 5 |
| True HIGH | 0 | 0 | 550 |

Per-class performance:
- LOW: F1 **1.000** (precision 1.000, recall 1.000, support 413)
- MODERATE: F1 **0.990** (precision 1.000, recall 0.980, support 251)
- HIGH: F1 **0.995** (precision 0.991, recall 1.000, support 550)

Figures: `reports/figures/confusion_matrix.png`, `confusion_matrix_normalized.png`, `feature_importance.png`, `class_distribution.png`, `learning_curve.png`, `depth_curve.png`.

## App

`streamlit run app/streamlit_app.py`

The demo is **FloodSense** (tagline: urban flood-risk intelligence). Pages: Dashboard, Predict, Batch analysis, Model insights, Dataset, About. Screenshots are in `docs/screenshots/`.

A footer stays on screen: academic prediction system, not an emergency-warning or evacuation tool. The classification-table banner stays on every page. If `models/model.json` is missing, the page tells you to run the two scripts. It does not train on startup.

Values outside the training range are refused. Presets load the training-set median of a class. A prediction shows the class, a confidence gauge, probabilities, and how many trees voted for the class.

## UI design decisions

- The product name in the app is FloodSense. The coursework title is unchanged in the report.
- Navigation is a custom sidebar. `st.navigation(..., position="hidden")` still registers the pages. The built-in list cannot hold the logo, the data chip, and the theme toggle together.
- Light is the default. Dark is a sidebar toggle. CSS variables restyle the shell; Plotly reads the same choice. The choice is kept in session state because a page change can reset the radio.
- Charts in the app are Plotly. Matplotlib files in `reports/figures/` are still what the report uses.
- Sliders use the training min and max. The number box still accepts the wider physical range, and a value outside the training range is refused with an inline alert. It is not clipped into a prediction.
- Preset buttons snap to the input step, so 99.15 mm of rainfall is shown as 99.
- The source banner is always visible when the table is not measured. The × mark is a label, not a control that hides the warning.
- Batch scoring draws a map only when the CSV has latitude and longitude.
- Key-driver percentiles use the seeded training split of `data/processed/dataset.csv`.
- The forest explorer shows one saved tree, top 3 levels, read only.
- Tables are HTML so risk badges and dark mode stay consistent.

## Tests

`python -m pytest -q` covers impurity math, the best split, depth and leaf limits, forest determinism, bootstrap draws, probability sums, out-of-bag range, metric values against scikit-learn, imputation fit on the training split only, save/load, a small end-to-end train that beats a majority-class baseline, and a smoke run of each FloodSense page (`tests/test_app.py`).

## Reproducibility

`config.RANDOM_STATE` is 42. Splits, bootstrap draws, and feature subsets use `numpy.random.RandomState` seeded from that value. The saved forest stores thresholds as hex floats so a reload predicts the same labels. Pinned packages are in `requirements.txt` (NumPy 2.1.3, pandas 2.2.3, Matplotlib 3.9.2, Streamlit 1.40.2, Plotly 5.24.1, pytest 8.3.4, scikit-learn 1.5.2).

## Limitations

The classifier is a coursework model. It is not a flood warning system. A real deployment would need validated operational data, spatial and temporal checks, uncertainty estimates, and review by people who work on floods. The accuracy above is on the coursework physical urban simulation, not a surveyed city. Feature importance does not show what causes flooding. Live rainfall, maps, sensors, alerts, and a mobile app are out of scope.

## Assumptions

- The active training dataset is the calibrated physical urban simulator in `scripts/generate_synthetic_data.py`. The historical disaster table `flood_dataset_classification.csv` is preserved in the repository root for comparison.
- IQR clipping is off. Rainfall above 2,000 mm and elevation below −20 m are set to missing and kept, with an indicator. A rainfall value repeated on more than 1% of the training rows is treated the same way. The repeated value is detected on the training split only.
- Physical range checks use the fixed bounds in `config.FEATURE_RANGES`. They are not learned from the sample.
- If a real file has no `risk_class` and a 0–1 hazard score, labels use fixed cuts at 0.33 and 0.66. If that score is `historical_flood`, the column is removed from the features. If the score is not on 0–1, `train.py` bins it with quantiles 0.40 and 0.75 fit on the training split only, then drops it.
- `elevation_relative_to_water` is the product of elevation and distance, because a water-surface elevation is not in the schema.
- `cumulative_rainfall` is created only when `rainfall_antecedent` exists.
- The structure search fits 15-tree forests on group folds of the training rows and picks the best macro-F1. `n_estimators` and class weight are compared on those same folds. The spatial test set is not used to choose the model. Macro-F1 is the objective because it treats the three classes equally; quadratic weighted kappa is reported beside it.
- `n_jobs` is accepted and ignored. Trees are fit one after another.
- Tree building is recursive and capped at depth 30.
- If two classes tie on votes, the lower class index wins. The order is LOW, MODERATE, HIGH.
- The app rejects out-of-range inputs. It does not clip them and still predict.
- Permutation importance uses at most 800 holdout rows and 3 shuffles. It is not used to pick hyperparameters.
- The page does not train a model on startup.

## Experiment log 2

External rainfall is CHIRPS v2.0, joined on latitude, longitude, and year. WorldPop was not downloaded (`python scripts/fetch_external.py --worldpop` would fetch it; GPW v4 is the login-gated fallback). Population columns are therefore missing, not imputed guesses at download time. The question that can be answered here is whether CHIRPS rainfall adds signal beyond latitude and longitude. The previous headline on the 5° spatial split was accuracy 0.470, macro F1 0.459, QWK 0.242. Numbers below are filled from `models/metrics.json` and `reports/ablation_external.csv` after this run. The table is the classification file plus those external rasters, not a surveyed flood record.

CHIRPS covers flood years 1981–2016 (3,686 of 4,192 rows, 87.9%). Of those, 3,510 (83.7%) also sit inside 50°S–50°N. After sampling, `rain_missing` is 16.5%. File rainfall and CHIRPS agree on the rows that can be compared: Pearson 0.799 on 2,542 rows after the filled value and values above 2,000 mm are set aside. That comparison uses the full flood table. The model still refits the filled-value list on the training split only. WorldPop years 2000–2020 would cover 2,474 rows (59.0%), but those rasters were not downloaded. `pop_missing` is 1 on every row. `pop_year_clipped` is 1 on 41.0% of rows and only marks years outside 2000–2020. It is not a population measurement. Rainfall before 1981 is missing. Details, licences, and the window rule are in `docs/dataset_documentation.md`.

The comparison holds the spatial cells fixed. Population B (year ≥ 2000 and both external sources present) has 0 rows, so exposure was not tested. On population A (all 4,186 rows), five spatial seeds with the forest seed held at 42 collapse to two partitions: seeds 42, 7, and 99 share the 837-row holdout, and seeds 11 and 123 share an 842-row holdout. Means below are still computed on the five runs.

| Feature set | Accuracy | Macro F1 | QWK |
| --- | ---: | ---: | ---: |
| Latitude and longitude (reference) | 0.473 ± 0.004 | 0.451 ± 0.011 | 0.254 ± 0.014 |
| Reference plus CHIRPS | 0.461 ± 0.008 | 0.445 ± 0.005 | 0.240 ± 0.002 |
| Reference plus population columns | 0.470 ± 0.004 | 0.450 ± 0.013 | 0.251 ± 0.023 |
| Full set (headline features) | 0.463 ± 0.009 | 0.446 ± 0.004 | 0.242 ± 0.001 |
| CHIRPS and population flags, no latitude or longitude | 0.415 ± 0.008 | 0.339 ± 0.008 | 0.084 ± 0.004 |

Paired differences, full set minus the latitude/longitude reference: accuracy −0.010 ± 0.013, macro F1 −0.004 ± 0.015, QWK −0.011 ± 0.015. On the headline seed the accuracy difference is 0. The set without latitude and longitude is −0.058 ± 0.004 accuracy versus the reference, and its accuracy (0.415) sits on the majority baseline (0.417). CHIRPS does not add signal beyond latitude and longitude, and it does not replace them. The population row is the year-clip flag plus columns that are missing everywhere. It is not an exposure result.

A 15° cell holdout scores accuracy 0.504, macro F1 0.463, QWK 0.323. That is 0.035 accuracy above the 5° headline, because the held-out regions changed, not because the model became more robust. Leaving out one continent at a time drops the row-weighted accuracy to 0.397 (a drop of 0.073). Asia is 0.311 accuracy on 1,482 test rows. Africa is 0.356 on 914. Europe's accuracy is 0.576 with macro F1 0.244 and QWK 0.000.

## Experiment log

- `time` is a four-digit year from 1900 to 2016 (94 distinct values, every value has four digits). It is not an epoch and not `YYYYMM`. Month and day are absent, so the model uses year only.
- Coordinates span about 53°S to 70°N and 172°W to 178°E. A 5° grid has 484 cells on the flood rows (median 5 rows, largest 71). The headline split holds out whole cells so a cell cannot sit in both train and test. Class balance is best effort: group assignment may use score tertiles only as a balance key, and the reported cuts are refit on the training rows.
- Dropping 767 high-rainfall rows left 3,417 rows and a random-split accuracy of 0.388, below a 0.407 majority baseline. Keeping those rows as missing, plus latitude, longitude, and year, leaves 4,186 rows. The spatial test accuracy is 0.470 against a 0.417 majority baseline.
- Ablation on that same spatial split: rainfall + elevation + slope 0.357; plus latitude and longitude 0.460; plus year 0.453; plus missingness indicators 0.470. Coordinates account for the lift. Rainfall's holdout permutation importance is about zero.
- The label is people affected, which tracks exposure and reporting, not water depth. The file cites no agency. Rainfall contains a filled value (1,383.125626 mm). There is no population layer. A later join of population density (WorldPop or GPW) and a gauge or reanalysis rainfall (CHIRPS or ERA5) on latitude, longitude, and year would test whether the coordinate effect is just where people were recorded.
