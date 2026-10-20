"""Decision tree, forest, save/load, and a small end-to-end train."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from flood_rf.decision_tree import DecisionTree, impurity
from flood_rf.features import FeatureEngineer
from flood_rf.persistence import FloodModel, load_model, save_model
from flood_rf.preprocessing import Preprocessor
from flood_rf.random_forest import RandomForestClassifier
from flood_rf.visualize import plot_confusion_matrix
from generate_synthetic_data import generate
from train import train_dataframe


def _depth(node) -> int:
    if node.feature is None:
        return 0
    return 1 + max(_depth(node.left), _depth(node.right))


def test_gini_and_entropy_match_hand_values():
    assert impurity(np.array([3, 1]), "gini") == pytest.approx(0.375)
    assert impurity(np.array([1, 1]), "gini") == pytest.approx(0.5)
    assert impurity(np.array([4, 0]), "gini") == pytest.approx(0.0)
    assert impurity(np.array([3, 1]), "entropy") == pytest.approx(0.8112781244591328)
    assert impurity(np.array([1, 1]), "entropy") == pytest.approx(1.0)
    with pytest.raises(ValueError):
        impurity(np.array([1, 1]), "other")


def test_best_split_finds_the_obvious_threshold():
    X = np.array([[0.0], [0.0], [0.0], [1.0], [1.0], [1.0]])
    y = np.array([0, 0, 0, 1, 1, 1])
    tree = DecisionTree(max_features=None, random_state=0).fit(X, y)
    assert tree.root_.feature == 0
    assert tree.root_.threshold == pytest.approx(0.5)
    assert np.array_equal(tree.predict(X), y)
    assert tree.feature_importances_[0] == pytest.approx(1.0)


def test_tree_fits_a_depth_two_pattern_and_respects_limits():
    X = np.array([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
    y = np.array([0, 0, 0, 1])
    shallow = DecisionTree(max_depth=1, max_features=None, random_state=0).fit(X, y)
    deep = DecisionTree(max_depth=2, max_features=None, random_state=0).fit(X, y)
    assert _depth(shallow.root_) <= 1
    assert accuracy(shallow, X, y) < 1.0
    assert _depth(deep.root_) <= 2
    assert np.array_equal(deep.predict(X), y)
    blocked = DecisionTree(min_samples_leaf=4, max_features=None, random_state=0)
    blocked.fit(np.array([[0.0], [1.0], [2.0], [3.0], [4.0], [5.0]]), np.array([0, 0, 0, 1, 1, 1]))
    assert blocked.root_.feature is None
    unsplittable = DecisionTree(min_samples_split=7, max_features=None, random_state=0)
    unsplittable.fit(np.array([[0.0], [1.0], [2.0], [3.0], [4.0], [5.0]]), np.array([0, 0, 0, 1, 1, 1]))
    assert unsplittable.root_.feature is None


def accuracy(model, X, y) -> float:
    return float(np.mean(model.predict(X) == y))


def test_sample_weight_of_ones_matches_unweighted():
    rng = np.random.RandomState(0)
    X = rng.normal(size=(30, 2))
    y = (X[:, 0] > 0).astype(np.int64)
    plain = DecisionTree(max_depth=3, max_features=None, random_state=1).fit(X, y)
    weighted = DecisionTree(max_depth=3, max_features=None, random_state=1).fit(X, y, sample_weight=np.ones(len(y)))
    assert np.array_equal(plain.predict(X), weighted.predict(X))


def test_entropy_tree_separates_the_same_threshold_problem():
    X = np.array([[0.0], [0.0], [1.0], [1.0]])
    y = np.array([0, 0, 1, 1])
    tree = DecisionTree(criterion="entropy", max_features=None, random_state=0).fit(X, y)
    assert np.array_equal(tree.predict(X), y)


def test_forest_votes_probabilities_oob_and_determinism():
    X = np.array([[0.0], [0.0], [0.0], [1.0], [1.0], [1.0]])
    y = np.array([0, 0, 0, 1, 1, 1])
    forest = RandomForestClassifier(
        n_estimators=5,
        max_features=None,
        bootstrap=False,
        oob_score=False,
        random_state=0,
    ).fit(X, y)
    votes = forest.predict_votes(X)
    assert np.all(votes[y == 0, 0] == 5)
    assert np.allclose(forest.predict_proba(X).sum(axis=1), 1.0)
    assert forest.feature_importances_.sum() == pytest.approx(1.0)

    rng = np.random.RandomState(0)
    Xn = rng.normal(size=(60, 4))
    yn = (Xn[:, 0] + Xn[:, 1] > 0).astype(np.int64)
    first = RandomForestClassifier(
        n_estimators=6, max_depth=4, max_features="sqrt", random_state=0, oob_score=True
    ).fit(Xn, yn)
    second = RandomForestClassifier(
        n_estimators=6, max_depth=4, max_features="sqrt", random_state=0, oob_score=True
    ).fit(Xn, yn)
    assert np.array_equal(first.predict(Xn), second.predict(Xn))
    assert first.oob_score_ == pytest.approx(second.oob_score_)
    assert 0.0 <= first.oob_score_ <= 1.0
    assert any(
        not np.array_equal(first.bootstrap_indices_[0], first.bootstrap_indices_[i])
        for i in range(1, len(first.bootstrap_indices_))
    )
    balanced = RandomForestClassifier(
        n_estimators=4, max_depth=3, class_weight="balanced", oob_score=False, random_state=1
    ).fit(Xn, yn)
    assert np.allclose(balanced.predict_proba(Xn).sum(axis=1), 1.0)
    with pytest.raises(ValueError):
        RandomForestClassifier(oob_score=True, bootstrap=False)


def test_save_load_roundtrip_predictions(tmp_path):
    frame = pd.DataFrame(
        {
            "rainfall": [10.0, 10.0, 80.0, 80.0],
            "impervious_area": [10.0, 12.0, 80.0, 90.0],
        }
    )
    y = np.array([0, 0, 1, 1])
    pre = Preprocessor().fit(frame)
    engineered = FeatureEngineer().transform(pre.transform(frame))
    names = ["rainfall", "impervious_area", "rainfall_x_impervious"]
    forest = RandomForestClassifier(
        n_estimators=3, max_depth=2, max_features=None, bootstrap=False, oob_score=False, random_state=0
    ).fit(engineered[names].to_numpy(dtype=np.float64), y, n_classes=2)
    model = FloodModel(
        forest,
        pre,
        names,
        ["LOW", "HIGH"],
        ["rainfall", "impervious_area"],
        {
            "input_ranges": {"rainfall": [10.0, 80.0], "impervious_area": [10.0, 90.0]},
            "defaults": {"rainfall": 10.0, "impervious_area": 12.0},
            "presets": {},
        },
    )
    path = tmp_path / "model.json"
    save_model(model, path)
    loaded = load_model(path)
    original = model.prepare(frame)
    assert np.array_equal(forest.predict(original), loaded.forest.predict(loaded.prepare(frame)))
    assert loaded.prepare(frame)[0, 2] == pytest.approx(1.0)
    result = loaded.predict_record({"rainfall": 80.0, "impervious_area": 90.0})
    assert result["risk_class"] in {"LOW", "HIGH"}
    assert sum(result["probabilities"].values()) == pytest.approx(1.0)
    with pytest.raises(ValueError):
        loaded.predict_record({"rainfall": 500.0, "impervious_area": 90.0})


def test_confusion_figure_is_written(tmp_path):
    plot_confusion_matrix([[2, 1], [0, 3]], ["LOW", "HIGH"], tmp_path / "cm.png")
    assert (tmp_path / "cm.png").stat().st_size > 0


def test_project_sources_do_not_import_sklearn():
    root = Path(__file__).resolve().parents[1]
    for folder in (root / "src", root / "app"):
        for path in folder.rglob("*.py"):
            _assert_no_sklearn_import(path)
    for path in (root / "scripts").glob("*.py"):
        if path.name == "benchmark_vs_sklearn.py":
            continue
        _assert_no_sklearn_import(path)


def _assert_no_sklearn_import(path: Path) -> None:
    for line in path.read_text(encoding="utf-8").lower().splitlines():
        stripped = line.strip()
        if stripped.startswith("import ") or stripped.startswith("from "):
            assert "sklearn" not in stripped, path


def test_end_to_end_beats_majority_baseline():
    frame = generate(n=700, seed=0)
    result = train_dataframe(frame, search=False, explain=False, n_estimators_final=12, verbose=False)
    assert result["metrics"]["data_source"] == "SYNTHETIC"
    assert result["metrics"]["test"]["accuracy"] > result["metrics"]["baseline_accuracy"] + 0.05
    assert result["metrics"]["test"]["f1_macro"] > 0.0
