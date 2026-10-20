"""Metric checks, including a comparison that is allowed to import sklearn."""

import numpy as np
import pytest
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score

from flood_rf.metrics import accuracy, classification_report, evaluate


LABELS = ["LOW", "MODERATE", "HIGH"]


def test_hand_computed_scores_and_zero_division():
    y_true = ["LOW", "HIGH"]
    y_pred = ["LOW", "LOW"]
    stats = evaluate(y_true, y_pred, LABELS)
    assert stats["accuracy"] == pytest.approx(0.5)
    assert stats["confusion_matrix"] == [[1, 0, 0], [0, 0, 0], [1, 0, 0]]
    assert stats["per_class"]["HIGH"]["precision"] == 0.0
    assert stats["per_class"]["HIGH"]["recall"] == 0.0
    assert stats["per_class"]["HIGH"]["f1"] == 0.0
    assert stats["per_class"]["MODERATE"]["support"] == 0
    assert "HIGH" in classification_report(y_true, y_pred, LABELS)


def test_matches_sklearn_on_an_uneven_example():
    y_true = ["LOW", "LOW", "MODERATE", "HIGH", "HIGH", "MODERATE", "LOW", "HIGH", "LOW", "MODERATE"]
    y_pred = ["LOW", "MODERATE", "MODERATE", "HIGH", "LOW", "HIGH", "LOW", "HIGH", "LOW", "LOW"]
    stats = evaluate(y_true, y_pred, LABELS)
    assert stats["accuracy"] == pytest.approx(accuracy_score(y_true, y_pred))
    assert stats["accuracy"] == pytest.approx(accuracy(y_true, y_pred))
    kwargs = {"labels": LABELS, "average": "macro", "zero_division": 0}
    assert stats["precision_macro"] == pytest.approx(precision_score(y_true, y_pred, **kwargs))
    assert stats["recall_macro"] == pytest.approx(recall_score(y_true, y_pred, **kwargs))
    assert stats["f1_macro"] == pytest.approx(f1_score(y_true, y_pred, **kwargs))
    weighted = {"labels": LABELS, "average": "weighted", "zero_division": 0}
    assert stats["precision_weighted"] == pytest.approx(precision_score(y_true, y_pred, **weighted))
    assert stats["f1_weighted"] == pytest.approx(f1_score(y_true, y_pred, **weighted))
    assert stats["confusion_matrix"] == confusion_matrix(y_true, y_pred, labels=LABELS).tolist()


def test_empty_prediction_accuracy_is_zero():
    assert accuracy([], []) == 0.0
    assert evaluate([], [], LABELS)["n"] == 0
