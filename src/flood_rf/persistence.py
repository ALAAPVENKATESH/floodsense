"""Save and load the forest, the preprocessor, and the feature schema.

Floats are stored as hex so a reload reproduces the same splits.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

import config
from flood_rf.explain import common_path_features, ranked_importance
from flood_rf.features import FeatureEngineer
from flood_rf.preprocessing import Preprocessor
from flood_rf.random_forest import RandomForestClassifier


def _freeze(value):
    if isinstance(value, dict):
        return {str(key): _freeze(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_freeze(item) for item in value]
    if isinstance(value, np.ndarray):
        return _freeze(value.tolist())
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, (float, np.floating)):
        number = float(value)
        if not math.isfinite(number):
            return None
        return {"__f__": float(number).hex()}
    if isinstance(value, int):
        return value
    if value is None or isinstance(value, str):
        return value
    raise TypeError(f"Cannot serialize {type(value)!r}")


def _hook(value):
    if isinstance(value, dict) and set(value) == {"__f__"}:
        return float.fromhex(value["__f__"])
    return value


class FloodModel:
    """Inference wrapper: validate raw inputs, impute, engineer, then vote."""

    def __init__(self, forest, preprocess, feature_names, class_labels, base_features, extra):
        self.forest = forest
        self.preprocess = preprocess
        self.feature_names = list(feature_names)
        self.class_labels = list(class_labels)
        self.base_features = list(base_features)
        self.extra = extra

    def validate(self, record: dict) -> list[str]:
        errors = []
        training_ranges = self.extra.get("input_ranges", {})
        for feature in self.base_features:
            raw = record.get(feature)
            if raw is None or (isinstance(raw, float) and math.isnan(raw)):
                errors.append(f"{feature} is required")
                continue
            try:
                value = float(raw)
            except (TypeError, ValueError):
                errors.append(f"{feature} must be a number")
                continue
            if feature in config.FEATURE_RANGES:
                lo, hi = config.FEATURE_RANGES[feature]
                if value < lo or value > hi:
                    errors.append(f"{feature}={value} is outside the allowed range [{lo}, {hi}]")
                    continue
            if feature in training_ranges:
                lo, hi = training_ranges[feature]
                if value < lo or value > hi:
                    errors.append(
                        f"{feature}={value} is outside the training range [{lo}, {hi}]. Prediction refused."
                    )
        return errors

    def prepare(self, frame: pd.DataFrame) -> np.ndarray:
        filled = self.preprocess.transform(frame)
        engineered = FeatureEngineer().transform(filled)
        missing = [name for name in self.feature_names if name not in engineered.columns]
        if missing:
            raise ValueError(f"Missing features after engineering: {missing}")
        return engineered[self.feature_names].to_numpy(dtype=np.float64)

    def predict_frame(self, frame: pd.DataFrame):
        matrix = self.prepare(frame)
        votes = self.forest.predict_votes(matrix)
        probabilities = self.forest.predict_proba(matrix)
        predicted = np.argmax(votes, axis=1)
        return predicted, probabilities, votes

    def predict_record(self, record: dict) -> dict:
        errors = self.validate(record)
        if errors:
            raise ValueError("; ".join(errors))
        frame = pd.DataFrame([{feature: float(record[feature]) for feature in self.base_features}])
        predicted, probabilities, votes = self.predict_frame(frame)
        index = int(predicted[0])
        label = self.class_labels[index]
        vote_row = votes[0]
        proba_row = probabilities[0]
        return {
            "risk_class": label,
            "probabilities": {name: float(proba_row[i]) for i, name in enumerate(self.class_labels)},
            "votes": {name: int(vote_row[i]) for i, name in enumerate(self.class_labels)},
            "n_trees": int(self.forest.n_estimators),
            "trees_for_predicted": int(vote_row[index]),
            "top_features": ranked_importance(self.feature_names, self.forest.feature_importances_)[:8],
            "path_features": common_path_features(self.forest, frame_to_row(self, frame), self.feature_names),
        }


def frame_to_row(model: FloodModel, frame: pd.DataFrame) -> np.ndarray:
    return model.prepare(frame)[0]


def save_model(model: FloodModel, path: Path | None = None) -> Path:
    path = config.MODEL_PATH if path is None else Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = _freeze(
        {
            "format": 1,
            "implementation": "flood_rf.RandomForestClassifier",
            "forest": model.forest.to_dict(),
            "preprocess": model.preprocess.to_dict(),
            "feature_names": model.feature_names,
            "class_labels": model.class_labels,
            "base_features": model.base_features,
            "extra": model.extra,
        }
    )
    path.write_text(json.dumps(payload, separators=(",", ":")), encoding="utf-8")
    return path


def load_model(path: Path | None = None) -> FloodModel:
    path = config.MODEL_PATH if path is None else Path(path)
    payload = json.loads(path.read_text(encoding="utf-8"), object_hook=_hook)
    forest = RandomForestClassifier.from_dict(payload["forest"])
    preprocess = Preprocessor.from_dict(payload["preprocess"])
    return FloodModel(
        forest,
        preprocess,
        payload["feature_names"],
        payload["class_labels"],
        payload["base_features"],
        payload.get("extra") or {},
    )
