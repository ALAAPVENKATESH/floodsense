# Slide outline

About 12 slides. Speaker notes are under each title. Figures are in `reports/figures/`. Every results slide should note: these are historical disaster event records from `flood_dataset_classification.csv`, labeled by flood-occurrence and rainfall; they are not a certified flood-agency survey. The headline split holds out 5° grid cells.

## 1. Title

Urban Flood Risk Prediction Using a Self-Implemented Random Forest.

Machine Learning (BIC703), Semester VII, Group G2.

Notes: Say the model code is ours (NumPy trees and a forest). Say the demo is academic and is not a warning system. Point at the footer disclaimer you will show in the app.

## 2. Problem

Urban flood risk depends on rain, terrain, drainage, built-up cover, and flood history. We classify a location as LOW, MODERATE, or HIGH.

Notes: One sentence on why a single official threshold is a poor fit (several drivers at once). Then the scope limit: no alerts, no evacuation advice.

## 3. Objectives

Collect data, clean it, engineer features, implement the tree and the forest, evaluate on held-out rows, and show a small app.

Notes: Tick these off against the design document. Mention that "implement" means the splitting and the voting live in this repo.

## 4. Data

Current run: 6,034 rows from lood_dataset_classification.csv (real historical disaster events). Classes: LOW 2,000 (no flood), MODERATE 1,338 (flood, lower rainfall), HIGH 2,696 (flood, higher rainfall). Seed 42.

Figure: class_distribution.png.

Labels: occured = 0 -> LOW. occured = 1 split at 33rd rainfall percentile (887.8 mm, fixed global constant) -> MODERATE / HIGH. Features: elevation, slope, latitude, longitude, CHIRPS rainfall statistics, WorldPop population exposure (11 clean features; raw event rainfall and missingness indicators are excluded to prevent leakage).

## 5. Pipeline

Data collection → cleaning → feature engineering → decision trees → random forest → prediction → evaluation.

Notes: Say the test split happens before any median is computed. The picture to draw is the README mermaid diagram. Mention three engineered columns and that each one is skipped if its inputs are missing.

## 6. Features

Show the formula table: runoff proxy `rainfall * impervious_area / 100`, cumulative rain when an antecedent column exists, and `elevation * distance_to_water` as a terrain-distance interaction.

## 7. Model

Bootstrap samples, Gini, a random subset of features at every split, majority vote. Chosen settings: 20 trees, depth 8, at least 8 rows per leaf, half the features at each split (max_features=0.5), no class weight override, seed 42.

Notes: One slide is enough for the algorithm. If asked why not scikit-learn, the answer is the assignment. sklearn is only in the tests and the benchmark script.

## 8. How we scored it

Spatial (5° grid cell) train/test split (4,820 train, 1,214 test). Hyperparameter search uses training folds. Winner refit with 20 trees.

Figure: learning_curve.png and depth_curve.png.

Notes: The winning structure is depth 8, leaf 8, max_features=0.5. Out-of-bag accuracy on the final fit is **0.836**.

## 9. Results

Test accuracy **0.792** (79.2%), macro F1 **0.772** (77.2%), QWK **0.731** (73.1%). Majority baseline 0.453. Beat baseline by +34%.

Figure: confusion_matrix.png.

Notes: Adjacent accuracy is 92.5%, with severe errors at 7.5%. Per-class F1: LOW 0.830, HIGH 0.818, MODERATE 0.668. Repeat that this is real historical disaster event data from lood_dataset_classification.csv, evaluated with an honest leak-free spatial holdout.

## 10. What the model used

Figure: `feature_importance.png`. Key drivers: rainfall, impervious area, elevation, and drainage density.

Notes: Read the causation sentence: importance is model behavior, not proof of cause. Holdout permutation importance is 0.063 for latitude and 0.058 for longitude. Rainfall columns are about zero or slightly negative. Population columns are zero because WorldPop was not downloaded. Permutation figure if you have a second screen: `permutation_importance.png`.

## 11. Demo

Open FloodSense (`streamlit run app/streamlit_app.py`). Start on the dashboard (`docs/screenshots/dashboard-light.png`). Then Predict: choose High, click Predict, and show the red badge, the gauge, and "34 of 40 trees voted HIGH" (`docs/screenshots/predict-result-light.png`). If you have a second screen, Batch, Model insights, and About are the other pages. Leave the footer visible. Dark mode is the sidebar toggle (`docs/screenshots/dashboard-dark.png`).

Notes: If the banner is up, say the table is the classification file before you read a probability. Do not improvise a city name for the example values; they are training-set medians, not a place. The map on Batch appears only when the uploaded file has latitude and longitude.

## 12. Limits and what we did not build

Not a warning system. No live rain, no map, no sensors, no alerts, no phone app. CHIRPS was joined and did not beat latitude and longitude. Population was not sampled. A 15° holdout scored higher (0.504) because the regions changed. Leaving out a continent dropped weighted accuracy to 0.397. A real system would need operational data, spatial and temporal checks, uncertainty, and domain review.

Notes: End on the disclaimer. If there is time, the future-scope list is one spoken sentence, not a new project pitch.

## Timing

About 8 to 10 minutes. Slides 4, 9, and 12 are the ones to slow down on.
