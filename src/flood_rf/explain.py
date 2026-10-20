"""Impurity importance, permutation importance, and per-prediction paths.

Importance describes how the fitted forest used each column. It is not evidence
that the column causes flooding.
"""

from __future__ import annotations

import numpy as np

import config
from flood_rf.metrics import accuracy

CAUSATION_NOTE = config.CAUSATION_NOTE


def ranked_importance(names, values) -> list[dict]:
    values = np.asarray(values, dtype=np.float64)
    order = np.argsort(-values, kind="mergesort")
    return [{"feature": str(names[i]), "importance": float(values[i])} for i in order]


def permutation_importance(model, X, y, feature_names, n_repeats: int = 3, random_state: int = 42, max_rows: int = 800) -> list[dict]:
    """Mean drop in accuracy when a column is shuffled. Not used for model selection.

    Held-out rows are capped at `max_rows` so the demo stays within a laptop budget.
    """
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y)
    rng = np.random.RandomState(random_state)
    if len(X) > max_rows:
        take = rng.choice(len(X), size=max_rows, replace=False)
        X = X[take]
        y = y[take]
    baseline = accuracy(y, model.predict(X))
    drops = np.zeros(X.shape[1], dtype=np.float64)
    for feature in range(X.shape[1]):
        scores = []
        for _ in range(n_repeats):
            shuffled = X.copy()
            rng.shuffle(shuffled[:, feature])
            scores.append(baseline - accuracy(y, model.predict(shuffled)))
        drops[feature] = float(np.mean(scores))
    return ranked_importance(feature_names, drops)


def common_path_features(forest, row, feature_names, top_k: int = 5) -> list[dict]:
    """Features that show up most often on the path taken through each tree."""
    counts: dict[str, int] = {}
    for tree in forest.trees_:
        for index in tree.decision_path_features(row):
            name = str(feature_names[index])
            counts[name] = counts.get(name, 0) + 1
    ranked = sorted(counts.items(), key=lambda item: (-item[1], item[0]))
    return [{"feature": name, "trees": n} for name, n in ranked[:top_k]]
