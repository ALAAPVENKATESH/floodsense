"""Bootstrap aggregation of the project decision trees. NumPy only.

Predictions are majority votes. Probabilities are the mean of the trees'
leaf distributions. Out-of-bag accuracy uses rows left out of each bootstrap
sample. `n_jobs` is accepted and ignored: trees are fit one after another.
"""

from __future__ import annotations

import numpy as np

from flood_rf.decision_tree import DecisionTree


class RandomForestClassifier:
    def __init__(
        self,
        n_estimators: int = 40,
        max_depth: int | None = 12,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        max_features="sqrt",
        criterion: str = "gini",
        bootstrap: bool = True,
        class_weight: str | None = None,
        oob_score: bool = False,
        n_jobs: int | None = 1,
        random_state: int | None = None,
        verbose: bool = False,
    ):
        if n_estimators < 1:
            raise ValueError("n_estimators must be >= 1")
        if class_weight not in (None, "balanced"):
            raise ValueError("class_weight must be None or 'balanced'")
        if oob_score and not bootstrap:
            raise ValueError("oob_score requires bootstrap=True")
        self.n_estimators = int(n_estimators)
        self.max_depth = max_depth
        self.min_samples_split = int(min_samples_split)
        self.min_samples_leaf = int(min_samples_leaf)
        self.max_features = max_features
        self.criterion = criterion
        self.bootstrap = bool(bootstrap)
        self.class_weight = class_weight
        self.oob_score = bool(oob_score)
        # ponytail: sequential fit. A process pool is worth it only if one tree exceeds ~1s.
        self.n_jobs = 1 if n_jobs is None else int(n_jobs)
        self.random_state = random_state
        self.verbose = verbose
        self.trees_: list[DecisionTree] = []
        self.bootstrap_indices_: list[np.ndarray] = []
        self.feature_importances_: np.ndarray | None = None
        self.oob_score_: float | None = None
        self.learning_curve_: list[tuple[int, float]] = []
        self.n_features_: int | None = None
        self.n_classes_: int | None = None

    def _class_weights(self, y: np.ndarray, sample_weight) -> np.ndarray:
        n = len(y)
        if self.class_weight is None:
            weights = np.ones(n, dtype=np.float64)
        else:
            counts = np.bincount(y, minlength=self.n_classes_).astype(np.float64)
            counts[counts == 0] = 1.0
            weights = (n / (self.n_classes_ * counts))[y]
        if sample_weight is not None:
            extra = np.asarray(sample_weight, dtype=np.float64)
            if len(extra) != n:
                raise ValueError("sample_weight must have one value per row")
            weights = weights * extra
        return weights

    def fit(self, X, y, sample_weight=None, n_classes: int | None = None) -> "RandomForestClassifier":
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.int64)
        if X.ndim != 2:
            raise ValueError("X must be 2-dimensional")
        if len(X) != len(y):
            raise ValueError("X and y must have the same number of rows")
        if len(y) == 0:
            raise ValueError("Cannot fit a forest on zero rows")
        n_samples = len(y)
        self.n_features_ = X.shape[1]
        self.n_classes_ = int(y.max()) + 1 if n_classes is None else int(n_classes)
        weights = self._class_weights(y, sample_weight)
        master = np.random.RandomState(self.random_state)
        seed_cap = np.iinfo(np.int32).max
        boot_seeds = master.randint(0, seed_cap, size=self.n_estimators)
        tree_seeds = master.randint(0, seed_cap, size=self.n_estimators)

        self.trees_ = []
        self.bootstrap_indices_ = []
        oob_sum = np.zeros((n_samples, self.n_classes_), dtype=np.float64)
        oob_count = np.zeros(n_samples, dtype=np.float64)
        curve: list[tuple[int, float]] = []
        step = max(1, self.n_estimators // 8)

        for i in range(self.n_estimators):
            draw = np.random.RandomState(int(boot_seeds[i]))
            if self.bootstrap:
                indices = draw.randint(0, n_samples, size=n_samples)
                in_bag = np.zeros(n_samples, dtype=bool)
                in_bag[indices] = True
                oob = ~in_bag
            else:
                indices = np.arange(n_samples)
                oob = np.zeros(n_samples, dtype=bool)
            self.bootstrap_indices_.append(indices)
            tree = DecisionTree(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                min_samples_leaf=self.min_samples_leaf,
                max_features=self.max_features,
                criterion=self.criterion,
                random_state=int(tree_seeds[i]),
            )
            tree.fit(
                X[indices],
                y[indices],
                sample_weight=weights[indices],
                n_classes=self.n_classes_,
            )
            self.trees_.append(tree)
            if self.oob_score and np.any(oob):
                oob_sum[oob] += tree.predict_proba(X[oob])
                oob_count[oob] += 1.0
                if (i + 1) % step == 0 or i + 1 == self.n_estimators:
                    valid = oob_count > 0
                    pred = np.argmax(oob_sum[valid], axis=1)
                    curve.append((i + 1, float(np.mean(pred == y[valid]))))
            if self.verbose and ((i + 1) % 10 == 0 or i + 1 == self.n_estimators):
                print(f"  tree {i + 1}/{self.n_estimators}", flush=True)

        stacked = np.vstack([tree.feature_importances_ for tree in self.trees_])
        mean_importance = stacked.mean(axis=0)
        total = float(mean_importance.sum())
        self.feature_importances_ = mean_importance / total if total > 0 else mean_importance
        self.learning_curve_ = curve
        if self.oob_score:
            valid = oob_count > 0
            if np.any(valid):
                pred = np.argmax(oob_sum[valid], axis=1)
                self.oob_score_ = float(np.mean(pred == y[valid]))
            else:
                self.oob_score_ = float("nan")
        else:
            self.oob_score_ = None
        return self

    def _as_2d(self, X) -> np.ndarray:
        X = np.asarray(X, dtype=np.float64)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        if X.shape[1] != self.n_features_:
            raise ValueError(f"Expected {self.n_features_} features, got {X.shape[1]}")
        return X

    def predict_votes(self, X) -> np.ndarray:
        """Raw vote counts, shape (n_rows, n_classes). Ties keep the smaller class index."""
        X = self._as_2d(X)
        votes = np.zeros((len(X), self.n_classes_), dtype=np.int64)
        rows = np.arange(len(X))
        for tree in self.trees_:
            votes[rows, tree.predict(X)] += 1
        return votes

    def predict(self, X) -> np.ndarray:
        return np.argmax(self.predict_votes(X), axis=1)

    def predict_proba(self, X) -> np.ndarray:
        X = self._as_2d(X)
        total = np.zeros((len(X), self.n_classes_), dtype=np.float64)
        for tree in self.trees_:
            total += tree.predict_proba(X)
        return total / self.n_estimators

    def to_dict(self) -> dict:
        return {
            "implementation": "flood_rf.RandomForestClassifier",
            "params": {
                "n_estimators": self.n_estimators,
                "max_depth": self.max_depth,
                "min_samples_split": self.min_samples_split,
                "min_samples_leaf": self.min_samples_leaf,
                "max_features": self.max_features,
                "criterion": self.criterion,
                "bootstrap": self.bootstrap,
                "class_weight": self.class_weight,
                "oob_score": self.oob_score,
                "n_jobs": self.n_jobs,
                "random_state": self.random_state,
            },
            "n_features": self.n_features_,
            "n_classes": self.n_classes_,
            "feature_importances": self.feature_importances_,
            "oob_score_": self.oob_score_,
            "learning_curve": self.learning_curve_,
            "trees": [tree.to_dict() for tree in self.trees_],
        }

    @classmethod
    def from_dict(cls, payload: dict) -> "RandomForestClassifier":
        params = dict(payload["params"])
        forest = cls(**params)
        forest.n_features_ = int(payload["n_features"])
        forest.n_classes_ = int(payload["n_classes"])
        forest.feature_importances_ = np.asarray(payload["feature_importances"], dtype=np.float64)
        oob = payload.get("oob_score_")
        forest.oob_score_ = None if oob is None else float(oob)
        forest.learning_curve_ = [tuple(point) for point in payload.get("learning_curve", [])]
        forest.trees_ = [DecisionTree.from_dict(tree) for tree in payload["trees"]]
        forest.bootstrap_indices_ = []
        return forest
