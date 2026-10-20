# AGENTS.md

Instructions for coding agents working in FloodSense.

## What this is

Coursework app for Machine Learning (BIC703), group G2. It classifies an urban location as LOW, MODERATE, or HIGH flood risk. The product name in the UI is FloodSense. The saved model is a random forest implemented in this repo with NumPy.

This is an academic demo. It is not an emergency-warning or evacuation system. Do not describe the scores as operational flood guidance.

## Hard rules

- Do not replace `src/flood_rf/decision_tree.py` or `src/flood_rf/random_forest.py` with scikit-learn, XGBoost, or any other ready-made tree or ensemble. Those files are the assignment.
- scikit-learn may appear only in `tests/` and `scripts/benchmark_vs_sklearn.py`.
- Do not invent measured rainfall, terrain, or flood history. Real tables belong in `data/raw/` as files the user supplies. If that folder has no CSV, the pipeline must keep using the labeled synthetic generator.
- Synthetic output stays named `flood_synthetic.csv`, with `data_source = SYNTHETIC`, a console warning, the in-app banner, and the README section. Never present those metrics as real-world performance.
- Fit preprocessing and the hyperparameter search on training data only. Score the held-out test set once.
- Keep `config.RANDOM_STATE` at 42 unless the user asks for a new experiment. A new seed means new metrics, and the README numbers must be updated from `models/metrics.json`.
- Do not commit `.env`, credentials, or `.venv/`.

## Layout

- `config.py` — seeds, paths, feature names, ranges, model defaults.
- `src/flood_rf/` — tree, forest, metrics, preprocessing, features, explain, plots, save/load.
- `scripts/build_dataset.py` — raw CSVs, or the synthetic fallback.
- `scripts/train.py` — search, one test evaluation, `models/`, `reports/figures/`.
- `app/` — Streamlit UI only. Do not retrain from the app.
- `docs/` — dataset notes, report outline, slides, screenshots.

## Commands

From the project root, with the virtual environment:

```powershell
.\.venv\Scripts\python.exe scripts\build_dataset.py
.\.venv\Scripts\python.exe scripts\train.py
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m streamlit run app\streamlit_app.py
```

Python 3.10 or newer. Dependencies are pinned in `requirements.txt`.

## Data and labels

- Empty `data/raw/` means synthetic. Do not download India-WRIS or Bhuvan data unless the user asks.
- If a real file already has `risk_class`, keep it. If the only label is a 0–1 hazard score, the documented cuts are 0.33 and 0.66. Other scores are binned on the training split only.
- `elevation_relative_to_water` is elevation times distance, not a surveyed water height.
- The app refuses inputs outside the training range. It does not clip them and still predict.

## UI

- Pages live in `app/views/`. Shared HTML and Plotly live in `app/ui/`.
- Matplotlib figures under `reports/figures/` stay the report assets. The app charts are Plotly.
- Keep the footer disclaimer and the synthetic banner. The banner is not dismissible.
- Theme choice is stored so a page change does not silently revert to light mode.

## When you change the model

Retrain, then replace the numbers in `README.md`, `docs/report_outline.md`, and `docs/slides_outline.md` from `models/metrics.json`. Say whether the table was synthetic.
