import numpy as np
import torch

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    balanced_accuracy_score,
    average_precision_score,
)


def calculate_metrics(y_true, y_pred, threshold=0.5):
    """
    Calculate classification metrics.

    Args:
        y_true: Ground truth labels (Tensor/ ndarray)
        y_pred: Prediction probabilities/ logits
        threshold: Decision threshold for converting predicted probabilities to class labels.

    Returns:
        dict
    """

    if isinstance(y_true, torch.Tensor):
        y_true = y_true.detach().cpu().numpy()

    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.detach().cpu().numpy()

    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)

    # convert to prob if input is logits
    if np.any(y_pred < 0) or np.any(y_pred > 1):
        y_pred = 1.0 / (1.0 + np.exp(-y_pred))

    y_label = (y_pred >= threshold).astype(np.int32)

    metrics = {
        "accuracy": accuracy_score(y_true, y_label),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_label),
        "precision": precision_score(y_true, y_label, zero_division=0),
        "recall": recall_score(y_true, y_label, zero_division=0),
        "f1": f1_score(y_true, y_label, zero_division=0),
    }

    try:
        metrics["auc"] = roc_auc_score(y_true, y_pred)
        metrics["ap"] = average_precision_score(y_true, y_pred)
    except ValueError:
        metrics["auc"] = 0.5
        metrics["ap"] = float(np.mean(y_true))

    return metrics


class AverageMeter:
    """Track running average."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0.0
        self.sum = 0.0
        self.count = 0
        self.avg = 0.0

    def update(self, val, n=1):
        self.val = float(val)
        self.sum += float(val) * n
        self.count += n
        self.avg = self.sum / self.count if self.count > 0 else 0.0
