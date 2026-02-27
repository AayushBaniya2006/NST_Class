import pytest
import numpy as np
from src.evaluation.metrics import compute_all_metrics


class TestComputeAllMetrics:
    def test_perfect_predictions(self):
        y_true = [0, 1, 2, 0, 1, 2]
        y_pred = [0, 1, 2, 0, 1, 2]
        y_proba = np.eye(3)[[0, 1, 2, 0, 1, 2]]
        metrics = compute_all_metrics(y_true, y_pred, y_proba, class_names=["12", "34", "56"])
        assert metrics["accuracy"] == 1.0
        assert metrics["macro_f1"] == 1.0

    def test_per_class_metrics_present(self):
        y_true = [0, 0, 1, 1, 2, 2]
        y_pred = [0, 1, 1, 1, 2, 0]
        y_proba = np.random.rand(6, 3)
        metrics = compute_all_metrics(y_true, y_pred, y_proba, class_names=["12", "34", "56"])
        assert "per_class" in metrics
        assert "12" in metrics["per_class"]
        assert "precision" in metrics["per_class"]["12"]
        assert "recall" in metrics["per_class"]["12"]

    def test_confusion_matrix_shape(self):
        y_true = [0, 1, 2, 0, 1, 2]
        y_pred = [0, 1, 2, 1, 0, 2]
        y_proba = np.random.rand(6, 3)
        metrics = compute_all_metrics(y_true, y_pred, y_proba, class_names=["12", "34", "56"])
        assert metrics["confusion_matrix"].shape == (3, 3)
