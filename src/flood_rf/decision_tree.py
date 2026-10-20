"""CART decision tree. NumPy only. No scikit-learn trees.

Splits are midpoints between sorted unique values of a numeric feature.
At every split the candidate features are a random subset (`max_features`).
Impurity uses sample weights when they are provided, so a forest can pass
balanced class weights. `min_samples_*` constraints use row counts.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

# ponytail: recursive build, depth capped at 30. Use an explicit stack if you need deeper trees.
HARD_MAX_DEPTH = 30


def impurity(counts: np.ndarray, criterion: str = "gini") -> float:
    """Impurity of one node from class counts (or weighted class masses)."""
    counts = np.asarray(counts, dtype=np.float64).ravel()
    total = float(counts.sum())
    if total <= 0:
        return 0.0
    p = counts / total
    if criterion == "gini":
        return float(1.0 - np.dot(p, p))
    if criterion == "entropy":
        p = p[p > 0]
        return float(-np.dot(p, np.log2(p)))
    raise ValueError(f"criterion must be 'gini' or 'entropy', got {criterion!r}")


def _impurity_rows(counts: np.ndarray, criterion: str) -> np.ndarray:
    totals = counts.sum(axis=1, keepdims=True)
    p = counts / np.maximum(totals, 1e-12)
    if criterion == "gini":
        return 1.0 - np.sum(p * p, axis=1)
    logp = np.zeros_like(p)
    np.log2(p, out=logp, where=p > 0)
    return -np.sum(p * logp, axis=1)


def _feature_subset(max_features, n_features: int, rng: np.random.RandomState) -> np.ndarray:
    if max_features is None:
        k = n_features
    elif isinstance(max_features, str):
        if max_features == "sqrt":
            k = max(1, int(math.sqrt(n_features)))
        elif max_features == "log2":
            k = max(1, int(math.log2(n_features))) if n_features > 1 else 1
        else:
            raise ValueError(f"Unknown max_features: {max_features!r}")
    elif isinstance(max_features, float):
        if not 0 < max_features <= 1:
            raise ValueError("float max_features must be in (0, 1]")
        k = max(1, int(max_features * n_features))
    else:
        k = int(max_features)
        if k <= 0:
            raise ValueError("max_features must be positive")
    k = min(k, n_features)
    if k == n_features:
        return np.arange(n_features)
    # Sort so equal-gain ties break toward the smaller feature index.
    return np.sort(rng.choice(n_features, size=k, replace=False))


class _Node:
    __slots__ = (
        "feature",
        "threshold",
        "left",
        "right",
        "value",
        "impurity",
        "n_samples",
        "weighted_n",
    )

    def __init__(self, value, impurity_value, n_samples, weighted_n, feature=None, threshold=None, left=None, right=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value
        self.impurity = impurity_value
        self.n_samples = n_samples
        self.weighted_n = weighted_n


def _node_to_dict(node: _Node) -> dict:
    payload: dict[str, Any] = {
        "value": np.asarray(node.value, dtype=np.float64),
        "impurity": float(node.impurity),
        "n_samples": int(node.n_samples),
        "weighted_n": float(node.weighted_n),
    }
    if node.feature is not None:
        payload["feature"] = int(node.feature)
        payload["threshold"] = float(node.threshold)
        payload["left"] = _node_to_dict(node.left)
        payload["right"] = _node_to_dict(node.right)
    return payload


def _node_from_dict(payload: dict) -> _Node:
    left = _node_from_dict(payload["left"]) if "left" in payload else None
    right = _node_from_dict(payload["right"]) if "right" in payload else None
    feature = payload.get("feature")
    return _Node(
        value=np.asarray(payload["value"], dtype=np.float64),
        impurity_value=float(payload["impurity"]),
        n_samples=int(payload["n_samples"]),
        weighted_n=float(payload["weighted_n"]),
        feature=None if feature is None else int(feature),
        threshold=None if "threshold" not in payload else float(payload["threshold"]),
        left=left,
        right=right,
    )


class DecisionTree:
    """Binary classification tree.

    Parameters
    ----------
    max_depth:
        Maximum number of splits on a path. ``None`` uses `HARD_MAX_DEPTH` (30).
    max_features:
        Features considered at each split: ``None`` (all), ``"sqrt"``, ``"log2"``,
        a positive int, or a float fraction in (0, 1].
    """

    def __init__(
        self,
        max_depth: int | None = None,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        max_features=None,
        criterion: str = "gini",
        random_state: int | None = None,
    ):
        if criterion not in ("gini", "entropy"):
            raise ValueError("criterion must be 'gini' or 'entropy'")
        if min_samples_split < 2:
            raise ValueError("min_samples_split must be >= 2")
        if min_samples_leaf < 1:
            raise ValueError("min_samples_leaf must be >= 1")
        self.max_depth = max_depth
        self.min_samples_split = int(min_samples_split)
        self.min_samples_leaf = int(min_samples_leaf)
        self.max_features = max_features
        self.criterion = criterion
        self.random_state = random_state
        self.root_: _Node | None = None
        self.n_features_: int | None = None
        self.n_classes_: int | None = None
        self.feature_importances_: np.ndarray | None = None
        self.rng_: np.random.RandomState | None = None

    def fit(self, X, y, sample_weight=None, n_classes: int | None = None) -> "DecisionTree":
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.int64)
        if X.ndim != 2:
            raise ValueError("X must be 2-dimensional")
        if len(X) != len(y):
            raise ValueError("X and y must have the same number of rows")
        if len(y) == 0:
            raise ValueError("Cannot fit a tree on zero rows")
        if np.isnan(X).any():
            raise ValueError("X contains NaN. Impute before fitting the tree.")
        if y.min() < 0:
            raise ValueError("y must be encoded as non-negative integers")
        n_samples, n_features = X.shape
        if sample_weight is None:
            sample_weight = np.ones(n_samples, dtype=np.float64)
        else:
            sample_weight = np.asarray(sample_weight, dtype=np.float64)
            if len(sample_weight) != n_samples:
                raise ValueError("sample_weight must have one value per row")
        self.n_features_ = n_features
        self.n_classes_ = int(y.max()) + 1 if n_classes is None else int(n_classes)
        if y.max() >= self.n_classes_:
            raise ValueError("y contains a class index >= n_classes")
        self.rng_ = np.random.RandomState(self.random_state)
        self.root_ = self._build(X, y, sample_weight, np.arange(n_samples), depth=0)
        self.feature_importances_ = self._importances()
        return self

    def _leaf(self, y_node: np.ndarray, w_node: np.ndarray) -> _Node:
        value = np.bincount(y_node, weights=w_node, minlength=self.n_classes_).astype(np.float64)
        return _Node(
            value=value,
            impurity_value=impurity(value, self.criterion),
            n_samples=int(len(y_node)),
            weighted_n=float(w_node.sum()),
        )

    def _build(self, X, y, sample_weight, indices: np.ndarray, depth: int) -> _Node:
        y_node = y[indices]
        w_node = sample_weight[indices]
        leaf = self._leaf(y_node, w_node)
        limit = HARD_MAX_DEPTH if self.max_depth is None else min(self.max_depth, HARD_MAX_DEPTH)
        if depth >= limit:
            return leaf
        if len(indices) < self.min_samples_split:
            return leaf
        if len(np.unique(y_node)) < 2:
            return leaf
        feature, threshold = self._best_split(X, y_node, w_node, indices)
        if feature is None:
            return leaf
        column = X[indices, feature]
        go_left = column <= threshold
        if go_left.all() or not go_left.any():
            return leaf
        left = self._build(X, y, sample_weight, indices[go_left], depth + 1)
        right = self._build(X, y, sample_weight, indices[~go_left], depth + 1)
        leaf.feature = int(feature)
        leaf.threshold = float(threshold)
        leaf.left = left
        leaf.right = right
        return leaf

    def _best_split(self, X, y_node, w_node, indices) -> tuple[int | None, float | None]:
        n = len(indices)
        parent = np.bincount(y_node, weights=w_node, minlength=self.n_classes_).astype(np.float64)
        parent_imp = impurity(parent, self.criterion)
        features = _feature_subset(self.max_features, self.n_features_, self.rng_)
        best_gain = 0.0
        best_feature = None
        best_threshold = None
        for feature in features:
            column = X[indices, int(feature)]
            if not np.isfinite(column).all():
                continue
            order = np.argsort(column, kind="mergesort")
            col_s = column[order]
            if col_s[0] == col_s[-1]:
                continue
            y_s = y_node[order]
            w_s = w_node[order]
            changes = np.flatnonzero(col_s[1:] != col_s[:-1])
            left_n = changes + 1
            right_n = n - left_n
            valid = (left_n >= self.min_samples_leaf) & (right_n >= self.min_samples_leaf)
            if not np.any(valid):
                continue
            changes = changes[valid]
            class_w = np.zeros((n, self.n_classes_), dtype=np.float64)
            class_w[np.arange(n), y_s] = w_s
            left = np.cumsum(class_w, axis=0)[changes]
            right = np.maximum(parent - left, 0.0)
            left_imp = _impurity_rows(left, self.criterion)
            right_imp = _impurity_rows(right, self.criterion)
            left_w = left.sum(axis=1)
            right_w = right.sum(axis=1)
            weighted = (left_w * left_imp + right_w * right_imp) / np.maximum(left_w + right_w, 1e-12)
            gain = parent_imp - weighted
            pick = int(np.argmax(gain))
            if gain[pick] > best_gain:
                best_gain = float(gain[pick])
                i = int(changes[pick])
                best_feature = int(feature)
                best_threshold = (float(col_s[i]) + float(col_s[i + 1])) / 2.0
        return best_feature, best_threshold

    def _importances(self) -> np.ndarray:
        scores = np.zeros(self.n_features_, dtype=np.float64)

        def walk(node: _Node) -> None:
            if node.feature is None:
                return
            decrease = (
                node.weighted_n * node.impurity
                - node.left.weighted_n * node.left.impurity
                - node.right.weighted_n * node.right.impurity
            )
            scores[node.feature] += decrease
            walk(node.left)
            walk(node.right)

        walk(self.root_)
        total = scores.sum()
        if total > 0:
            scores /= total
        return scores

    def _as_2d(self, X) -> np.ndarray:
        X = np.asarray(X, dtype=np.float64)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        if X.shape[1] != self.n_features_:
            raise ValueError(f"Expected {self.n_features_} features, got {X.shape[1]}")
        return X

    def predict_proba(self, X) -> np.ndarray:
        X = self._as_2d(X)
        out = np.zeros((len(X), self.n_classes_), dtype=np.float64)
        stack = [(self.root_, np.arange(len(X)))]
        while stack:
            node, idx = stack.pop()
            if len(idx) == 0:
                continue
            if node.feature is None:
                total = float(node.value.sum())
                if total <= 0:
                    out[idx] = 1.0 / self.n_classes_
                else:
                    out[idx] = node.value / total
                continue
            go_left = X[idx, node.feature] <= node.threshold
            stack.append((node.left, idx[go_left]))
            stack.append((node.right, idx[~go_left]))
        return out

    def predict(self, X) -> np.ndarray:
        return np.argmax(self.predict_proba(X), axis=1)

    def decision_path_features(self, x) -> list[int]:
        row = self._as_2d(x)[0]
        node = self.root_
        used = []
        while node is not None and node.feature is not None:
            used.append(int(node.feature))
            node = node.left if row[node.feature] <= node.threshold else node.right
        return used

    def to_dict(self) -> dict:
        return {
            "params": {
                "max_depth": self.max_depth,
                "min_samples_split": self.min_samples_split,
                "min_samples_leaf": self.min_samples_leaf,
                "max_features": self.max_features,
                "criterion": self.criterion,
                "random_state": self.random_state,
            },
            "n_features": self.n_features_,
            "n_classes": self.n_classes_,
            "feature_importances": self.feature_importances_,
            "root": _node_to_dict(self.root_),
        }

    @classmethod
    def from_dict(cls, payload: dict) -> "DecisionTree":
        params = payload["params"]
        tree = cls(
            max_depth=params["max_depth"],
            min_samples_split=params["min_samples_split"],
            min_samples_leaf=params["min_samples_leaf"],
            max_features=params["max_features"],
            criterion=params["criterion"],
            random_state=params["random_state"],
        )
        tree.n_features_ = int(payload["n_features"])
        tree.n_classes_ = int(payload["n_classes"])
        tree.feature_importances_ = np.asarray(payload["feature_importances"], dtype=np.float64)
        tree.root_ = _node_from_dict(payload["root"])
        return tree
