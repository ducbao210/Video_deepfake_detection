import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    roc_curve,
    auc as sk_auc,
    precision_recall_curve,
    average_precision_score,
    ConfusionMatrixDisplay,
)


# --------------------------------------------------------------------------- #
# History persistence
# --------------------------------------------------------------------------- #
def save_history(history: dict, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)


def load_history(path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# --------------------------------------------------------------------------- #
# Learning curves (train vs val, side by side subplots)
# --------------------------------------------------------------------------- #
def plot_learning_curves(history: dict, save_path=None, metrics=("loss", "auc")):
    """
    Args:
        history: {"epoch": [...], "train_loss": [...], "val_loss": [...],
                  "train_auc": [...], "val_auc": [...], ...}
        metrics: base metric names (without train_/val_ prefix) to plot,
                 one subplot per metric.
    """
    epochs = history.get("epoch", list(range(1, len(history["train_loss"]) + 1)))

    metrics = [m for m in metrics if f"train_{m}" in history or f"val_{m}" in history]
    n = max(len(metrics), 1)
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 4.5))
    axes = np.atleast_1d(axes)

    for ax, m in zip(axes, metrics):
        if f"train_{m}" in history:
            ax.plot(epochs, history[f"train_{m}"], marker="o", label=f"train_{m}")
        if f"val_{m}" in history:
            ax.plot(epochs, history[f"val_{m}"], marker="o", label=f"val_{m}")
        ax.set_xlabel("Epoch")
        ax.set_ylabel(m)
        ax.set_title(f"{m} over epochs")
        ax.grid(alpha=0.3)
        ax.legend()

    fig.tight_layout()
    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
    return fig


# --------------------------------------------------------------------------- #
# Confusion matrix
# --------------------------------------------------------------------------- #
def plot_confusion_matrix(
    cm, class_names=("Real", "Fake"), normalize=False, save_path=None
):
    cm = np.asarray(cm, dtype=float)
    if normalize:
        cm = cm / cm.sum(axis=1, keepdims=True)

    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    fig, ax = plt.subplots(figsize=(5, 5))
    disp.plot(
        ax=ax, cmap="Blues", values_format=".2f" if normalize else ".0f", colorbar=False
    )
    ax.set_title("Confusion Matrix" + (" (normalized)" if normalize else ""))

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
    return fig


# --------------------------------------------------------------------------- #
# ROC curve
# --------------------------------------------------------------------------- #
def plot_roc_curve(y_true, y_prob, save_path=None):
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = sk_auc(fpr, tpr)

    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    ax.plot(fpr, tpr, label=f"ROC (AUC = {roc_auc:.4f})")
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Random")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve")
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3)

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
    return fig


# --------------------------------------------------------------------------- #
# Precision-Recall curve
# --------------------------------------------------------------------------- #
def plot_pr_curve(y_true, y_prob, save_path=None):
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    ap = average_precision_score(y_true, y_prob)

    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    ax.plot(recall, precision, label=f"PR curve (AP = {ap:.4f})")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curve")
    ax.legend(loc="lower left")
    ax.grid(alpha=0.3)

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
    return fig
