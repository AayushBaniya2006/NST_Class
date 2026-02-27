import pytest
import numpy as np
from src.evaluation.metrics import compute_all_metrics

CLASS_NAMES = ["1", "2", "3", "4", "5", "6"]


class TestComputeAllMetrics:
    def test_perfect_predictions(self):
        y_true = [0, 1, 2, 3, 4, 5]
        y_pred = [0, 1, 2, 3, 4, 5]
        y_proba = np.eye(6)[[0, 1, 2, 3, 4, 5]]
        metrics = compute_all_metrics(y_true, y_pred, y_proba, class_names=CLASS_NAMES)
        assert metrics["accuracy"] == 1.0
        assert metrics["macro_f1"] == 1.0

    def test_per_class_metrics_present(self):
        y_true = [0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5]
        y_pred = [0, 1, 1, 1, 2, 0, 3, 4, 4, 4, 5, 3]
        y_proba = np.random.rand(12, 6)
        metrics = compute_all_metrics(y_true, y_pred, y_proba, class_names=CLASS_NAMES)
        assert "per_class" in metrics
        assert "1" in metrics["per_class"]
        assert "precision" in metrics["per_class"]["1"]
        assert "recall" in metrics["per_class"]["1"]

    def test_confusion_matrix_shape(self):
        y_true = [0, 1, 2, 3, 4, 5]
        y_pred = [0, 1, 2, 3, 4, 5]
        y_proba = np.random.rand(6, 6)
        metrics = compute_all_metrics(y_true, y_pred, y_proba, class_names=CLASS_NAMES)
        assert metrics["confusion_matrix"].shape == (6, 6)
