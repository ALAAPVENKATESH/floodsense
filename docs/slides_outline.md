# Slide outline

About 12 slides. Speaker notes are under each title. Figures are in `reports/figures/`. Every results slide needs the spoken line: these numbers are people-affected bins from `flood_dataset_classification.csv`, not a flood-agency survey. The headline split holds out 5° cells.

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

Current run: 4,186 flood-positive rows from `flood_dataset_classification.csv`, seed 42. Classes: LOW 1,738, MODERATE 1,401, HIGH 1,047.

Figure: `class_distribution.png`.

Source: the local file. Schema is in `data/raw/README.md`.

Notes: Say the file cites no agency. `time` is a year, 1900–2016. The label is people affected: at or below 3,000 is LOW, at or below 55,700 is MODERATE, otherwise HIGH. Those cuts were fit on the spatial training split only. The score is not given to the model.

## 5. Pipeline

Data collection → cleaning → feature engineering → decision trees → random forest → prediction → evaluation.

Notes: Say the test split happens before any median is computed. The picture to draw is the README mermaid diagram. Mention three engineered columns and that each one is skipped if its inputs are missing.

## 6. Features

Show the formula table: runoff proxy `rainfall * impervious_area / 100`, cumulative rain when an antecedent column exists, and `elevation * distance_to_water` as a stand-in because we do not have a water level.

Notes: Say the third formula is mixed units and is only a rank feature. Do not call it height above the river.

## 7. Model

Bootstrap samples, Gini, a random subset of features at every split, majority vote. Chosen settings: 40 trees, depth 8, at least 8 rows per leaf, half the features at each split, balanced class weights, seed 42.

Notes: One slide is enough for the algorithm. If asked "why not scikit-learn?", the answer is the assignment. sklearn is only in the tests and the benchmark script. If asked how a split is chosen: sort the feature, try midpoints, pick the largest impurity drop, with a vectorized class-count scan.

## 8. How we scored it

20% of rows, but whole 5° cells are held out together. The class cuts are fit on the training side only. Search uses group folds of the training rows. Refit the winner with 40 trees. Also show the random split and label it leakage-prone.

Figure: `learning_curve.png` and, if there is room, `depth_curve.png`.

Notes: The winning structure is depth 8, leaf 8, `max_features=1.0`. Forty trees without class weights beat balanced weights on the training folds (validation macro-F1 0.464 versus 0.453). Out-of-bag accuracy on the final fit is 0.482.

## 9. Results

Spatial accuracy 0.470, macro F1 0.449, QWK 0.243. Majority baseline 0.417. The previous feature set on the same cells is 0.470 / 0.459 / 0.242. Random-split accuracy 0.519 (optimistic). Five reruns: accuracy 0.461 ± 0.010.

Figure: `confusion_matrix.png`.

Notes: This beats the majority class. File rainfall, elevation, and slope alone score 0.421 accuracy and 0.236 macro F1 on the same cells. Latitude and longitude are what move the score. CHIRPS does not improve on them. Repeat that the classes are people-affected bins, not a surveyed depth.

## 10. What the model used

Figure: `feature_importance.png`. Largest impurity shares: longitude 0.224, latitude 0.214.

Notes: Read the causation sentence: importance is model behavior, not proof of cause. Holdout permutation importance is 0.063 for latitude and 0.058 for longitude. Rainfall columns are about zero or slightly negative. Population columns are zero because WorldPop was not downloaded. Permutation figure if you have a second screen: `permutation_importance.png`.

## 11. Demo

Open FloodSense (`streamlit run app/streamlit_app.py`). Start on the dashboard (`docs/screenshots/dashboard-light.png`). Then Predict: choose High, click Predict, and show the red badge, the gauge, and "34 of 40 trees voted HIGH" (`docs/screenshots/predict-result-light.png`). If you have a second screen, Batch, Model insights, and About are the other pages. Leave the footer visible. Dark mode is the sidebar toggle (`docs/screenshots/dashboard-dark.png`).

Notes: If the banner is up, say the table is the classification file before you read a probability. Do not improvise a city name for the example values; they are training-set medians, not a place. The map on Batch appears only when the uploaded file has latitude and longitude.

## 12. Limits and what we did not build

Not a warning system. No live rain, no map, no sensors, no alerts, no phone app. CHIRPS was joined and did not beat latitude and longitude. Population was not sampled. A 15° holdout scored higher (0.504) because the regions changed. Leaving out a continent dropped weighted accuracy to 0.397. A real system would need operational data, spatial and temporal checks, uncertainty, and domain review.

Notes: End on the disclaimer. If there is time, the future-scope list is one spoken sentence, not a new project pitch.

## Timing

About 8 to 10 minutes. Slides 4, 9, and 12 are the ones to slow down on.
