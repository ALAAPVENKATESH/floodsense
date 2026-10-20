# Slide outline

About 12 slides. Speaker notes are under each title. Figures are in `reports/figures/`. Every results slide needs the spoken line: these numbers are from the synthetic table, not from a flood agency.

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

Current run: 8,000 synthetic rows, seed 42. Classes are nearly even (LOW 2,747, MODERATE 2,655, HIGH 2,598).

Figure: `class_distribution.png`.

Real path, not used in these numbers: National Water Data Portal for rainfall, Bhuvan for geographic and flood layers. Schema is in `data/raw/README.md`.

Notes: Read the synthetic warning aloud. Explain the label in one sentence: a weighted risk score, cut at the 0.34 and 0.67 quantiles, then 6% of labels flipped in a cycle. The score column is not given to the model.

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

20% stratified holdout. Search on a validation slice of the training rows only (four candidates, 15 trees). Refit the winner with 40 trees. Report train and test.

Figure: `learning_curve.png` and, if there is room, `depth_curve.png`.

Notes: Out-of-bag accuracy rose from 0.558 at 5 trees to 0.604 at 40, so extra trees helped a little and then flattened. The depth-10 candidate did not win on validation macro-F1 (0.607 vs 0.615).

## 9. Results

Test accuracy 0.617, macro precision 0.632, macro recall 0.617, macro F1 0.622. Train accuracy 0.713. Gap 0.096. Majority baseline 0.343.

Figure: `confusion_matrix.png`.

Notes: Walk the matrix. LOW→HIGH is 35 and HIGH→LOW is 14, so the extreme classes are not often swapped. MODERATE precision is 0.474 because the middle bin overlaps both neighbours. Repeat that this is the simulator.

## 10. What the model used

Figure: `feature_importance.png`. Top shares: rainfall × impervious 0.241, drainage density 0.142, elevation × distance 0.107.

Notes: Read the causation sentence: importance is model behavior, not proof of cause. Mention that shuffling `rainfall` alone barely changes holdout accuracy because the engineered rainfall features already contain it. Permutation figure if you have a second screen: `permutation_importance.png`.

## 11. Demo

Open FloodSense (`streamlit run app/streamlit_app.py`). Start on the dashboard (`docs/screenshots/dashboard-light.png`). Then Predict: choose High, click Predict, and show the red badge, the gauge, and "34 of 40 trees voted HIGH" (`docs/screenshots/predict-result-light.png`). If you have a second screen, Batch, Model insights, and About are the other pages. Leave the footer visible. Dark mode is the sidebar toggle (`docs/screenshots/dashboard-dark.png`).

Notes: If the banner is up, say the word synthetic before you read a probability. Do not improvise a city name for the example values; they are training-set medians, not a place. The map on Batch appears only when the uploaded file has latitude and longitude.

## 12. Limits and what we did not build

Not a warning system. No live rain, no map, no sensors, no alerts, no phone app. A real system would need operational data, spatial and temporal checks, uncertainty, and domain review.

Notes: End on the disclaimer. If there is time, the future-scope list is one spoken sentence, not a new project pitch.

## Timing

About 8 to 10 minutes. Slides 4, 9, and 12 are the ones to slow down on.
