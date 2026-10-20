"""Classification metrics implemented with NumPy. Zero-division yields 0."""

from __future__ import annotations

from typing import Iterable

import numpy as np


def confusion_matrix(y_true, y_pred, labels: Iterable) -> np.ndarray:
    """Rows are true labels, columns are predicted labels, in `labels` order."""
    labels = list(labels)
    index = {label: i for i, label in enumerate(labels)}
    y_true = list(y_true)
    y_pred = list(y_pred)
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have the same length")
    yt = np.empty(len(y_true), dtype=np.int64)
    yp = np.empty(len(y_pred), dtype=np.int64)
    for i, (truth, pred) in enumerate(zip(y_true, y_pred)):
        if truth not in index or pred not in index:
            raise ValueError(f"Label outside {labels}: {truth!r}, {pred!r}")
        yt[i] = index[truth]
        yp[i] = index[pred]
    cm = np.zeros((len(labels), len(labels)), dtype=np.int64)
    np.add.at(cm, (yt, yp), 1)
    return cm


def _safe_div(num: np.ndarray, den: np.ndarray) -> np.ndarray:
    out = np.zeros_like(num, dtype=np.float64)
    np.divide(num, den, out=out, where=den != 0)
    return out


def per_class_scores(cm: np.ndarray) -> dict[str, np.ndarray]:
    cm = np.asarray(cm, dtype=np.float64)
    tp = np.diag(cm)
    support = cm.sum(axis=1)
    predicted = cm.sum(axis=0)
    precision = _safe_div(tp, predicted)
    recall = _safe_div(tp, support)
    f1 = _safe_div(2 * precision * recall, precision + recall)
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "support": support,
    }


def _weighted(values: np.ndarray, support: np.ndarray) -> float:
    support = np.asarray(support, dtype=np.float64)
    if support.sum() == 0:
        return 0.0
    return float(np.average(np.asarray(values, dtype=np.float64), weights=support))


def accuracy(y_true, y_pred) -> float:
    y_true = np.asarray(list(y_true))
    y_pred = np.asarray(list(y_pred))
    if len(y_true) == 0:
        return 0.0
    return float(np.mean(y_true == y_pred))


def evaluate(y_true, y_pred, labels: Iterable) -> dict:
    """Accuracy, per-class scores, macro/weighted averages, and the confusion matrix."""
    labels = list(labels)
    cm = confusion_matrix(y_true, y_pred, labels)
    scores = per_class_scores(cm)
    total = int(cm.sum())
    acc = float(np.trace(cm) / total) if total else 0.0
    per_class = {}
    for i, label in enumerate(labels):
        per_class[str(label)] = {
            "precision": float(scores["precision"][i]),
            "recall": float(scores["recall"][i]),
            "f1": float(scores["f1"][i]),
            "support": int(scores["support"][i]),
        }
    return {
        "accuracy": acc,
        "precision_macro": float(np.mean(scores["precision"])),
        "recall_macro": float(np.mean(scores["recall"])),
        "f1_macro": float(np.mean(scores["f1"])),
        "precision_weighted": _weighted(scores["precision"], scores["support"]),
        "recall_weighted": _weighted(scores["recall"], scores["support"]),
        "f1_weighted": _weighted(scores["f1"], scores["support"]),
        "per_class": per_class,
        "confusion_matrix": cm.astype(int).tolist(),
        "labels": [str(label) for label in labels],
        "n": total,
        "adjacent_accuracy": adjacent_accuracy(y_true, y_pred, labels),
        "quadratic_weighted_kappa": quadratic_weighted_kappa(y_true, y_pred, labels),
        "mean_absolute_class_error": mean_absolute_class_error(y_true, y_pred, labels),
        "severe_error_rate": severe_error_rate(y_true, y_pred, labels),
    }


def _label_codes(y_true, y_pred, labels: Iterable) -> tuple[np.ndarray, np.ndarray, dict]:
    labels = list(labels)
    index = {label: i for i, label in enumerate(labels)}
    yt = np.empty(len(list(y_true)), dtype=np.int64)
    yp = np.empty(len(list(y_pred)), dtype=np.int64)
    for i, (truth, pred) in enumerate(zip(y_true, y_pred)):
        if truth not in index or pred not in index:
            raise ValueError(f"Label outside {labels}: {truth!r}, {pred!r}")
        yt[i] = index[truth]
        yp[i] = index[pred]
    return yt, yp, index


def adjacent_accuracy(y_true, y_pred, labels: Iterable) -> float:
    """Share of predictions at most one class away from the truth."""
    y_true = list(y_true)
    y_pred = list(y_pred)
    if not y_true:
        return 0.0
    yt, yp, _index = _label_codes(y_true, y_pred, labels)
    return float(np.mean(np.abs(yt - yp) <= 1))


def mean_absolute_class_error(y_true, y_pred, labels: Iterable) -> float:
    """Mean |true class index − predicted class index|."""
    y_true = list(y_true)
    y_pred = list(y_pred)
    if not y_true:
        return 0.0
    yt, yp, _index = _label_codes(y_true, y_pred, labels)
    return float(np.mean(np.abs(yt - yp)))


def severe_error_rate(y_true, y_pred, labels: Iterable) -> float:
    """Share of rows where LOW is predicted HIGH or HIGH is predicted LOW."""
    y_true = list(y_true)
    y_pred = list(y_pred)
    if not y_true:
        return 0.0
    yt, yp, index = _label_codes(y_true, y_pred, labels)
    if "LOW" not in index or "HIGH" not in index:
        return 0.0
    low, high = index["LOW"], index["HIGH"]
    severe = ((yt == low) & (yp == high)) | ((yt == high) & (yp == low))
    return float(np.mean(severe))


def quadratic_weighted_kappa(y_true, y_pred, labels: Iterable) -> float:
    """Cohen's kappa with quadratic weights on the class order in `labels`."""
    y_true = list(y_true)
    y_pred = list(y_pred)
    labels = list(labels)
    if not y_true or len(labels) < 2:
        return 0.0
    yt, yp, _index = _label_codes(y_true, y_pred, labels)
    k = len(labels)
    observed = np.zeros((k, k), dtype=np.float64)
    np.add.at(observed, (yt, yp), 1.0)
    expected = np.outer(observed.sum(axis=1), observed.sum(axis=0)) / len(yt)
    scale = np.arange(k, dtype=np.float64)
    weights = ((scale[:, None] - scale[None, :]) / (k - 1)) ** 2
    denom = float(np.sum(weights * expected))
    if denom == 0.0:
        return 0.0
    return float(1.0 - np.sum(weights * observed) / denom)


def classification_report(y_true, y_pred, labels: Iterable) -> str:
    stats = evaluate(y_true, y_pred, labels)
    header = f"{'label':<12}{'precision':>10}{'recall':>10}{'f1':>10}{'support':>10}"
    lines = [header]
    for label in stats["labels"]:
        row = stats["per_class"][label]
        lines.append(
            f"{label:<12}{row['precision']:10.3f}{row['recall']:10.3f}"
            f"{row['f1']:10.3f}{row['support']:10d}"
        )
    lines.append(
        f"{'macro':<12}{stats['precision_macro']:10.3f}{stats['recall_macro']:10.3f}"
        f"{stats['f1_macro']:10.3f}{stats['n']:10d}"
    )
    lines.append(
        f"{'weighted':<12}{stats['precision_weighted']:10.3f}"
        f"{stats['recall_weighted']:10.3f}{stats['f1_weighted']:10.3f}{stats['n']:10d}"
    )
    lines.append(f"{'accuracy':<12}{stats['accuracy']:10.3f}")
    return "\n".join(lines)


def bootstrap_ci(y_true, y_pred, labels: Iterable, n_boot: int = 1000, seed: int = 42, alpha: float = 0.05) -> dict:
    """Percentile interval for holdout accuracy and macro F1. Resamples rows with replacement."""
    y_true = np.asarray(list(y_true))
    y_pred = np.asarray(list(y_pred))
    if len(y_true) == 0:
        return {"n_boot": 0, "accuracy": {"low": 0.0, "high": 0.0}, "f1_macro": {"low": 0.0, "high": 0.0}}
    rng = np.random.RandomState(int(seed))
    acc = np.empty(int(n_boot), dtype=np.float64)
    f1 = np.empty(int(n_boot), dtype=np.float64)
    n = len(y_true)
    for i in range(int(n_boot)):
        idx = rng.randint(0, n, n)
        stats = evaluate(y_true[idx], y_pred[idx], labels)
        acc[i] = stats["accuracy"]
        f1[i] = stats["f1_macro"]
    low_q, high_q = alpha / 2.0, 1.0 - alpha / 2.0
    return {
        "n_boot": int(n_boot),
        "seed": int(seed),
        "accuracy": {"low": float(np.quantile(acc, low_q)), "high": float(np.quantile(acc, high_q))},
        "f1_macro": {"low": float(np.quantile(f1, low_q)), "high": float(np.quantile(f1, high_q))},
    }
