# Raw data

Put measured CSV files in this folder and run `python scripts/build_dataset.py`.

If this folder has no CSV files, the build script writes a simulated dataset and labels it `SYNTHETIC`. Do not add the synthetic file here; that would make the next build treat it as a user-supplied table.

This project does not download data. The columns below are the ones the pipeline knows how to use. Extra columns are kept only when they are merge keys or a label.

## Expected columns

Names are case-insensitive. Spaces and hyphens become underscores. Aliases are listed in `config.py` (`COLUMN_ALIASES`).

| Column | Unit / scale | Role |
| --- | --- | --- |
| rainfall | mm over the chosen period | feature |
| rainfall_intensity | mm/h | feature |
| rainfall_antecedent | mm in the previous period | optional feature |
| elevation | metres | feature |
| slope | degrees | feature |
| drainage_density | km/km² | feature |
| impervious_area | percent built-up / impervious | feature |
| distance_to_water | km | feature |
| historical_flood | 0–1 indicator or score | feature, unless it is the only label source |
| flood_hazard_score | hazard score | label source only, not a model input |
| risk_class | LOW, MODERATE, or HIGH | label, if you already have one |
| location_id | text | merge key |
| district + period | text | used to build a merge key when `location_id` is absent |
| latitude + longitude + period | degrees, period id | used to build a merge key |

## How files are merged

- One CSV: it is used as the table.
- Several CSVs: they are inner-joined on `location_id` if every file has it. Otherwise `district` is used when every file has it. If files have `district` and `period` (or `latitude`, `longitude`, and `period`) but no id, an id is built from those fields first.
- If no shared key exists and the files have the same number of rows, columns are placed side by side in filename order. That assumption is recorded in `data/processed/dataset_meta.json`. Prefer a real key.

## Label rule

1. If `risk_class` is present, it is the label. Synonyms such as `medium` are mapped to `MODERATE`.
2. Else if `flood_hazard_score` is on a 0–1 scale, fixed cuts are used: `<= 0.33` LOW, `<= 0.66` MODERATE, otherwise HIGH. The score is not a feature.
3. Else if `historical_flood` is on a 0–1 scale, the same fixed cuts are used and that column is removed from the features so the label is not a copy of one input.
4. Else the score column is kept and `scripts/train.py` bins it with quantiles 0.40 and 0.75 fit on the training split only, then drops it from the features.

Rows outside the physical bounds in `config.FEATURE_RANGES` are dropped. Missing optional measurements stay missing; they are imputed inside the training pipeline from the training split only.

## Sources to cite when you add real files

Record the version, geographic coverage, time period, and licence in `docs/dataset_documentation.md` and in `dataset_meta.json` after the build.

- Rainfall observations: [India-WRIS / National Water Data Portal](https://indiawris.gov.in/).
- Geographic and flood layers: [ISRO / NRSC Bhuvan](https://bhuvan.nrsc.gov.in/).

Do not invent values for a station or a grid cell that the file does not contain. If a column is missing, the model trains without it.
