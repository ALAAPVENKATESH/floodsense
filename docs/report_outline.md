# Project report outline

Machine Learning (BIC703), Semester VII, Group G2.

Title: Urban Flood Risk Prediction Using a Self-Implemented Random Forest Algorithm.

The tables below are filled from `models/metrics.json` after `python scripts/train.py` on the synthetic 8,000-row table (seed 42). If you retrain on measured CSVs, replace every "SYNTHETIC" number and say so in section 7.

## 1. Project overview

An academic classifier that maps rainfall, terrain, drainage, land cover, and a historical flood indicator to LOW, MODERATE, or HIGH. The decision tree and the random forest are implemented in `src/flood_rf/` with NumPy. A library forest is not the saved model.

State in the introduction that the submitted run used simulated rows. See section 7.

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

For this run the model used all twelve columns named in `metrics.json` → `features`.

## 6. Output

LOW, MODERATE, HIGH. The app also shows class probabilities (mean of the trees' leaf distributions) and the vote count ("N of 40 trees voted HIGH"). A tie on votes goes to the earlier class in that order.

## 7. Dataset

**This run: SYNTHETIC.** 8,000 rows, seed 42. Class counts: LOW 2,747, MODERATE 2,655, HIGH 2,598. Figure: `reports/figures/class_distribution.png`.

No file was taken from the National Water Data Portal or Bhuvan. Those remain the intended sources if the group adds CSVs. Record version, coverage, period, and licence in `docs/dataset_documentation.md` before claiming a measured result.

Label rule for this table: latent score quantiles 0.34 and 0.67, then 6% cycled label noise. The score is not a column in the CSV.

## 8. Machine learning design

Describe, with the code as the reference:

- Bootstrap sample per tree, with a seed derived from 42
- Gini impurity
- At each split, a random feature subset (`max_features=0.5` in the chosen model, so 6 of 12 columns)
- Midpoint thresholds between sorted unique values
- Stop on depth, minimum samples, a pure node, or no split with positive gain
- Final prediction by majority vote
- Class weights `balanced`

Search on the training split only (15 trees each). Validation macro-F1:

| max_depth | min_samples_leaf | max_features | Validation macro-F1 |
| ---: | ---: | --- | ---: |
| 6 | 12 | sqrt | 0.609 |
| 8 | 8 | sqrt | 0.614 |
| 10 | 5 | sqrt | 0.607 |
| 8 | 8 | 0.5 | 0.615 |

The last row was selected. It was refit with 40 trees. Figure: `reports/figures/depth_curve.png`. Out-of-bag accuracy by number of trees: `reports/figures/learning_curve.png` (0.558 at 5 trees, 0.604 at 40).

## 9. Application design

Streamlit page with Prediction, Model performance, Feature importance, Dataset, and About. Inputs are checked against the training min and max. Invalid values produce an error and no prediction. Footer disclaimer on every view. Synthetic banner when `data_source` is not REAL.

Screenshots to paste: the prediction panel for a high-risk example, and the performance tab.

## 10. System modules

One short paragraph per module in `src/flood_rf/` plus `scripts/` and `app/streamlit_app.py`. Match the responsibility table in the project design document.

## 11. Technology stack

Python, NumPy 2.1.3, pandas 2.2.3, Matplotlib 3.9.2, Streamlit 1.40.2. scikit-learn 1.5.2 is installed for tests and the optional benchmark script only.

## 12. Evaluation

Held-out test, 1,600 rows (20%). Majority-class accuracy on that split: 0.343.

| Split | Accuracy | Macro precision | Macro recall | Macro F1 |
| --- | ---: | ---: | ---: | ---: |
| Train | 0.713 | 0.727 | 0.714 | 0.716 |
| Test | 0.617 | 0.632 | 0.617 | 0.622 |

Train minus test accuracy gap: 0.096. Out-of-bag accuracy: 0.604. Fit and search: 50.8 seconds.

Test per class:

| Class | Precision | Recall | F1 | Support |
| --- | ---: | ---: | ---: | ---: |
| LOW | 0.720 | 0.617 | 0.665 | 549 |
| MODERATE | 0.474 | 0.569 | 0.517 | 531 |
| HIGH | 0.703 | 0.665 | 0.684 | 520 |

Confusion matrix (true by predicted), figure `reports/figures/confusion_matrix.png`:

|  | LOW | MODERATE | HIGH |
| --- | ---: | ---: | ---: |
| LOW | 339 | 175 | 35 |
| MODERATE | 118 | 302 | 111 |
| HIGH | 14 | 160 | 346 |

Discuss the moderate class: it sits between the two score quantiles, so rainfall and elevation that are only slightly unusual fall here, and 6% label noise hits this boundary hardest. LOW and HIGH are rarely confused with each other (35 and 14 cells).

Write one sentence that these figures are simulator scores.

## 13. Explainability

Impurity importance, normalized to sum to 1, averaged across trees. Figure: `reports/figures/feature_importance.png`.

| Feature | Impurity share |
| --- | ---: |
| rainfall_x_impervious | 0.241 |
| drainage_density | 0.142 |
| elevation_relative_to_water | 0.107 |
| elevation | 0.100 |
| impervious_area | 0.086 |

Permutation importance on a held-out subsample (`reports/figures/permutation_importance.png`) puts the same three features first. `rainfall` alone is about zero because `rainfall_x_impervious` and `cumulative_rainfall` already use it.

Required sentence: feature importance reflects model behavior, not proof of causation.

The prediction panel also lists features that appear most often on the path taken through the trees for that row. That is a description of the vote, not a cause.

## 14. Non-functional requirements

- The interface is one page with five tabs and labeled inputs.
- The full training run finished in 50.8 seconds. A single prediction is a walk down 40 trees.
- Inputs outside the training range are rejected with a visible message.
- Rebuild with the commands in the README, the pinned requirements, seed 42, and `docs/dataset_documentation.md`.

## 15. Deliverables

| Deliverable | Path |
| --- | --- |
| Source | `src/flood_rf/`, `scripts/`, `app/` |
| Dataset | `data/processed/dataset.csv`, `flood_synthetic.csv`, `dataset_meta.json` |
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

- Why the forest beats a majority baseline (0.617 vs 0.343) and why that still is not a field validation.
- Why the train/test gap of about 0.10 is acceptable here, and which search row was slightly more overfit (depth 10, leaf 5, validation macro-F1 0.607).
- Why a self-implemented forest was required, and where scikit-learn is allowed to appear (tests and `scripts/benchmark_vs_sklearn.py` only).
