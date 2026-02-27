"""Evaluation metrics for skin tone classification."""
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
    roc_auc_score,
)


def compute_all_metrics(
    y_true: list,
    y_pred: list,
    y_proba: np.ndarray,
    class_names: list = None,
) -> dict:
    """Compute all evaluation metrics.

    Args:
        y_true: Ground truth labels.
        y_pred: Predicted labels.
        y_proba: Predicted probabilities (N x num_classes).
        class_names: Names for each class.

    Returns:
        Dict with accuracy, macro_f1, per_class metrics, confusion_matrix, roc_auc.
    """
    if class_names is None:
        class_names = ["1", "2", "3", "4", "5", "6"]

    num_classes = len(class_names)
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    accuracy = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro")

    per_class_precision = precision_score(y_true, y_pred, average=None, zero_division=0)
    per_class_recall = recall_score(y_true, y_pred, average=None, zero_division=0)
    per_class_f1 = f1_score(y_true, y_pred, average=None, zero_division=0)

    per_class = {}
    for i, name in enumerate(class_names):
        per_class[name] = {
            "precision": float(per_class_precision[i]),
            "recall": float(per_class_recall[i]),
            "f1": float(per_class_f1[i]),
            "support": int(np.sum(y_true == i)),
        }

    cm = confusion_matrix(y_true, y_pred, labels=list(range(num_classes)))

    try:
        roc_auc = roc_auc_score(y_true, y_proba, multi_class="ovr", average="macro")
        per_class_auc = roc_auc_score(y_true, y_proba, multi_class="ovr", average=None)
        for i, name in enumerate(class_names):
            per_class[name]["roc_auc"] = float(per_class_auc[i])
    except ValueError:
        roc_auc = None

    return {
        "accuracy": float(accuracy),
        "macro_f1": float(macro_f1),
        "roc_auc": float(roc_auc) if roc_auc is not None else None,
        "per_class": per_class,
        "confusion_matrix": cm,
    }
