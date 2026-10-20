"""Matplotlib figures for the report and the Streamlit app."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


CLASS_COLORS = {"LOW": "#2f6f4e", "MODERATE": "#8a5a12", "HIGH": "#8c2f2f"}


def _save(fig, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def plot_confusion_matrix(cm, labels, path: Path, normalize: bool = False) -> None:
    cm = np.asarray(cm, dtype=np.float64)
    shown = cm.copy()
    title = "Confusion matrix (counts)"
    if normalize:
        row_sums = cm.sum(axis=1, keepdims=True)
        shown = np.divide(cm, row_sums, out=np.zeros_like(cm), where=row_sums != 0)
        title = "Confusion matrix (row-normalized)"
    fig, ax = plt.subplots(figsize=(5.2, 4.4))
    image = ax.imshow(shown, cmap="Blues")
    fig.colorbar(image, ax=ax, fraction=0.046)
    ax.set_xticks(range(len(labels)), labels)
    ax.set_yticks(range(len(labels)), labels)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
    for i in range(shown.shape[0]):
        for j in range(shown.shape[1]):
            text = f"{shown[i, j]:.2f}" if normalize else f"{int(cm[i, j])}"
            ax.text(j, i, text, ha="center", va="center", color="black")
    _save(fig, path)


def plot_feature_importance(names, values, path: Path, title: str = "Feature importance") -> None:
    values = np.asarray(values, dtype=np.float64)
    names = np.asarray(names)
    order = np.argsort(values)
    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.barh(names[order], values[order], color="#1f4e79")
    ax.set_xlabel("Importance (sums to 1)" if "permutation" not in title.lower() else "Accuracy drop when shuffled")
    ax.set_title(title)
    _save(fig, path)


def plot_class_distribution(counts: dict, path: Path) -> None:
    labels = list(counts.keys())
    values = [counts[label] for label in labels]
    colors = [CLASS_COLORS.get(label, "#1f4e79") for label in labels]
    fig, ax = plt.subplots(figsize=(5, 3.6))
    ax.bar(labels, values, color=colors)
    ax.set_ylabel("Rows")
    ax.set_title("Class distribution")
    _save(fig, path)


def plot_learning_curve(curve, path: Path, ylabel: str = "Out-of-bag accuracy") -> None:
    if not curve:
        return
    xs = [point[0] for point in curve]
    ys = [point[1] for point in curve]
    fig, ax = plt.subplots(figsize=(5.4, 3.6))
    ax.plot(xs, ys, marker="o", color="#1f4e79")
    ax.set_xlabel("Trees")
    ax.set_ylabel(ylabel)
    ax.set_ylim(0, 1)
    ax.set_title("Learning curve")
    _save(fig, path)


def plot_depth_curve(rows: list[dict], path: Path) -> None:
    """Validation macro-F1 against max_depth from the search log. Not a one-factor experiment."""
    if len(rows) < 2:
        return
    best: dict[int, float] = {}
    for row in rows:
        depth = row["max_depth"]
        score = row["validation_macro_f1"]
        best[depth] = max(score, best.get(depth, 0.0))
    depths = sorted(best)
    fig, ax = plt.subplots(figsize=(5.4, 3.6))
    ax.plot(depths, [best[d] for d in depths], marker="o", color="#8a5a12")
    ax.set_xlabel("max_depth")
    ax.set_ylabel("Validation macro-F1")
    ax.set_ylim(0, 1)
    ax.set_title("Search: depth vs validation macro-F1")
    _save(fig, path)
