# Project report outline

Machine Learning (BIC703), Semester VII, Group G2.

Title: Urban Flood Risk Prediction Using a Self-Implemented Random Forest Algorithm.

The tables below are filled from `models/metrics.json` after `python scripts/train.py` on `flood_dataset_classification.csv` (4,186 rows, seed 42, spatial holdout). These are people-affected bins from that file, not surveyed flood performance.

## 1. Project overview

An academic classifier that maps rainfall, terrain, drainage, land cover, and a historical flood indicator to LOW, MODERATE, or HIGH. The decision tree and the random forest are implemented in `src/flood_rf/` with NumPy. A library forest is not the saved model.

State in the introduction that the submitted run used `flood_dataset_classification.csv`. See section 7.

## 2. Problem statement

Urban flooding depends on rainfall, terrain, drainage, land cover, and flood history. The system classifies one location. It does not issue warnings or evacuation advice. Quote the disclaimer in `config.DISCLAIMER`.

## 3. Objectives

Map each objective to a file:

| Objective | Where it is done |
| --- | --- |
| Collect and prepare data | `scripts/build_dataset.py`, `data/raw/README.md` |
| Clean and preprocess | `src/flood_rf/preprocessing.py` |
| Engineer features | `src/flood_rf/features.py` |
| Implement trees and a forest | `decision_tree.py`, `random_forest.py` |
| Train and evaluate on held-out rows | `scripts/train.py` |
| Accept location conditions and show the class | `app/streamlit_app.py` |

## 4. Proposed system

Use the pipeline figure from the README: data collection, cleaning, feature engineering, decision trees, random forest, prediction, evaluation and visualization.

Say what is fit on the training split only: medians, modes, category codes, optional IQR bounds (off by default), and the hyperparameter choice. The test set is scored once.

## 5. Input features

List the nine raw columns and the three engineered formulas from `docs/dataset_documentation.md`. Note that a column absent from a real CSV is dropped, not filled with a made-up measurement.

For this run the model used rainfall, elevation, slope, latitude, longitude, year, and the two missingness indicators.

## 6. Output

LOW, MODERATE, HIGH. The app also shows class probabilities (mean of the trees' leaf distributions) and the vote count ("N of 40 trees voted HIGH"). A tie on votes goes to the earlier class in that order.

## 7. Dataset

**This run: CLASSIFICATION.** 4,186 flood-positive rows from `flood_dataset_classification.csv`, seed 42. Class counts: LOW 1,738, MODERATE 1,401, HIGH 1,047. Figure: `reports/figures/class_distribution.png`.

Source: the local file. No agency is cited. Not India-WRIS and not Bhuvan. Other disaster types (2,045 rows) are excluded. They are not called LOW. `time` is a calendar year from 1900 to 2016. Coordinates cover the globe in 484 cells of 5°.

Label rule: people affected. Spatial-training quantiles 0.40 and 0.75 land at 3,000 and 55,700. LOW at or below 3,000, MODERATE at or below 55,700, otherwise HIGH. The score is not a column the forest sees. Rainfall equal to 1,383.125626 mm on the training split (13.0% of those rows), rainfall above 2,000 mm, and elevation below −20 m are missing values plus an indicator, not deleted rows.

## 8. Machine learning design

Describe, with the code as the reference:

- Bootstrap sample per tree, with a seed derived from 42
- Gini impurity
- At each split, a random feature subset (`max_features=0.5` in the chosen model)
- Midpoint thresholds between sorted unique values
- Stop on depth, minimum samples, a pure node, or no split with positive gain
- Final prediction by majority vote
- Class weights `balanced`

Search on the training split only (15 trees each). Validation macro-F1:

| max_depth | min_samples_leaf | max_features | Validation macro-F1 |
| ---: | ---: | --- | ---: |
| 6 | 12 | sqrt | 0.436 |
| 8 | 8 | sqrt | 0.433 |
| 10 | 5 | sqrt | 0.437 |
| 8 | 8 | log2 | 0.444 |
| 8 | 8 | 0.5 | 0.452 |
| 8 | 8 | 1.0 | 0.439 |

The search objective is macro-F1, so the three classes count equally. Quadratic weighted kappa is reported and was not the selection score. Depth 8, leaf 8, and `max_features=0.5` won. On that structure, 40 trees with balanced weights (validation macro-F1 0.459) beat 40 trees with no weights (0.451) and beat 20 trees. Figure: `reports/figures/depth_curve.png`. Out-of-bag accuracy on the final fit is 0.486.

## 9. Application design

FloodSense pages: Dashboard, Predict, Batch analysis, Model insights, Dataset, and About. Inputs are checked against the training min and max. Invalid values produce an error and no prediction. Footer disclaimer on every view. A classification-table banner stays up because `data_source` is not REAL.

Screenshots to paste: the prediction panel for a high-risk example, and the performance tab.

## 10. System modules

One short paragraph per module in `src/flood_rf/` plus `scripts/` and `app/streamlit_app.py`. Match the responsibility table in the project design document.

## 11. Technology stack

Python, NumPy 2.1.3, pandas 2.2.3, Matplotlib 3.9.2, Streamlit 1.40.2. scikit-learn 1.5.2 is installed for tests and the optional benchmark script only.

## 12. Evaluation

Headline holdout: 837 rows in cells the training split did not see. Same 3,349 / 837 cells as the previous headline. Majority-class accuracy on that split: 0.417. Stratified random guess: 0.331.

| Comparison | Accuracy | Macro F1 | QWK |
| --- | ---: | ---: | ---: |
| Previous 3-feature random split | 0.388 | 0.372 | — |
| Random split (optimistic, leakage-prone; includes year) | 0.519 | 0.512 | 0.343 |
| Spatial test (headline, with CHIRPS) | 0.470 | 0.449 | 0.243 |
| Previous feature set, same cells | 0.470 | 0.459 | 0.242 |
| Majority class | 0.417 | 0.196 | 0.000 |
| Stratified random guess | 0.331 | 0.320 | −0.019 |

The previous-feature row refits file rainfall, elevation, slope, latitude, longitude, year, and the two file missingness flags with the old hyperparameters (40 trees, depth 8, leaf 8, `max_features=0.5`, balanced weights). It matches the published 0.470 / 0.459 / 0.242. The new headline uses CHIRPS columns and the population flags. Population measurements are missing on every row.

Train accuracy on the spatial fit: 0.589. Train minus test accuracy gap: 0.120. Out-of-bag accuracy: 0.482. Adjacent accuracy 0.811. Mean absolute class error 0.719. Severe-error rate 0.189. Bootstrap 95% interval on the headline predictions: accuracy 0.436–0.504, macro F1 0.416–0.483.

Five reruns that reseed the split and the forest: accuracy 0.461 ± 0.010, macro F1 0.439 ± 0.018, QWK 0.242 ± 0.014.

Ablation under the chosen hyperparameters, same spatial split (`reports/ablation.csv`):

| Features | Accuracy | Macro F1 | QWK |
| --- | ---: | ---: | ---: |
| rainfall, elevation, slope | 0.421 | 0.236 | 0.032 |
| + latitude, longitude | 0.470 | 0.443 | 0.243 |
| + year | 0.468 | 0.452 | 0.236 |
| + missingness indicators (previous feature set) | 0.479 | 0.457 | 0.242 |

CHIRPS and the empty population columns are in `reports/ablation_external.csv` and the README experiment log. CHIRPS does not beat latitude and longitude. The set without coordinates scores 0.415 accuracy, on the majority baseline. Population was not sampled.

Confusion matrix (true by predicted), figure `reports/figures/confusion_matrix.png`:

|  | LOW | MODERATE | HIGH |
| --- | ---: | ---: | ---: |
| LOW | 205 | 53 | 91 |
| MODERATE | 130 | 69 | 79 |
| HIGH | 67 | 24 | 119 |

Spatial accuracy is above the majority baseline. File rainfall, elevation, and slope alone are not. The gain appears when latitude and longitude are added. Say that the label is people affected, so a coordinate can stand in for where people were recorded. These figures are not a surveyed city. A 15° holdout scores 0.504 accuracy, which is higher because the regions change. Leaving out one continent drops the row-weighted accuracy to 0.397.

## 13. Explainability

Impurity importance, normalized to sum to 1, averaged across trees. Figure: `reports/figures/feature_importance.png`.

| Feature | Impurity share |
| --- | ---: |
| longitude | 0.224 |
| latitude | 0.214 |
| elevation | 0.097 |
| rainfall_file | 0.086 |
| slope | 0.082 |
| rain_wet_month_share | 0.081 |
| rain_anomaly | 0.077 |
| rain_max_month_mm | 0.069 |
| rain_annual_mm | 0.059 |

Permutation importance on the spatial holdout is latitude 0.063 and longitude 0.058. Slope is 0.006 and elevation is 0.004. Every rainfall column is about zero or slightly negative (`rain_anomaly` −0.010). The three population measurements and `pop_missing` are exactly zero. `pop_year_clipped` is a year flag, not a count of people.

Required sentence: feature importance reflects model behavior, not proof of causation.

The prediction panel also lists features that appear most often on the path taken through the trees for that row. That is a description of the vote, not a cause.

## 14. Non-functional requirements

- The interface is one page with five tabs and labeled inputs.
- The full training run, including the comparison splits and five seeds, finished in about 13 minutes. A single prediction is a walk down 40 trees.
- Inputs outside the training range are rejected with a visible message.
- Rebuild with the commands in the README, the pinned requirements, seed 42, and `docs/dataset_documentation.md`.

## 15. Deliverables

| Deliverable | Path |
| --- | --- |
| Source | `src/flood_rf/`, `scripts/`, `app/` |
| Dataset | `flood_dataset_classification.csv`, `data/raw/flood_classification.csv`, `data/processed/dataset.csv`, `dataset_meta.json` |
| Dependencies | `requirements.txt` |
| Report | this outline, to be written up from these numbers |
| Demo | `streamlit run app/streamlit_app.py` |
| Slides | `docs/slides_outline.md` |
| Figures | `reports/figures/` |
| Saved model and metrics | `models/model.json`, `metrics.json`, `feature_importance.json` |

## 16. Future scope

Live rainfall, a map, water-level sensors, alerts, finer spatial prediction, a mobile client. None of these are in the submission.

## 17. Scope note

Close the report with the academic disclaimer. Say what would be required before any operational use: validated data, domain review, spatial and temporal validation, uncertainty, and a safety process. Do not describe the demo as a warning system.

## Suggested discussion points

- Spatial accuracy (0.470) beats the majority baseline (0.417). The previous feature set on the same cells is 0.470 accuracy and 0.459 macro F1. Adding CHIRPS keeps the accuracy and lowers macro F1 to 0.449. The random split (0.519) is higher because nearby records leak, and that split also includes year.
- The label is people affected. The file rainfall contains a filled value. File rainfall and CHIRPS correlate at 0.799 on 2,542 rows, and the model still does not use rainfall to lift the score. Latitude and longitude carry the model. WorldPop was not downloaded, so this run does not test exposure.
- CHIRPS does not beat latitude and longitude (paired accuracy −0.010 ± 0.013). Without coordinates, accuracy is 0.415. A coarser 15° holdout scores 0.504 because the regions change. Leaving out a continent drops weighted accuracy to 0.397.
- Why a self-implemented forest was required, and where scikit-learn is allowed to appear (tests and `scripts/benchmark_vs_sklearn.py` only).
