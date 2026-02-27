import pytest
from src.evaluation.fairness import compute_fairness_gap, compare_model_fairness


class TestFairnessGap:
    def test_zero_gap_when_equal(self):
        per_class = {
            "1": {"recall": 0.80},
            "2": {"recall": 0.80},
            "3": {"recall": 0.80},
            "4": {"recall": 0.80},
            "5": {"recall": 0.80},
            "6": {"recall": 0.80},
        }
        result = compute_fairness_gap(per_class)
        assert result["gap"] == 0.0
        assert not result["significant"]

    def test_detects_significant_gap(self):
        per_class = {
            "1": {"recall": 0.92},
            "2": {"recall": 0.90},
            "3": {"recall": 0.85},
            "4": {"recall": 0.75},
            "5": {"recall": 0.60},
            "6": {"recall": 0.50},
        }
        result = compute_fairness_gap(per_class, threshold=0.15)
        assert result["gap"] == pytest.approx(0.42)
        assert result["significant"]
        assert result["best_class"] == "1"
        assert result["worst_class"] == "6"

    def test_custom_threshold(self):
        per_class = {
            "1": {"recall": 0.80},
            "2": {"recall": 0.78},
            "3": {"recall": 0.75},
            "4": {"recall": 0.72},
            "5": {"recall": 0.70},
            "6": {"recall": 0.65},
        }
        result = compute_fairness_gap(per_class, threshold=0.10)
        assert result["significant"]


class TestCompareModelFairness:
    def test_comparison_table(self):
        models = {
            "EfficientNetV2": {
                "1": {"recall": 0.90, "precision": 0.88},
                "2": {"recall": 0.88, "precision": 0.85},
                "3": {"recall": 0.82, "precision": 0.80},
                "4": {"recall": 0.78, "precision": 0.75},
                "5": {"recall": 0.65, "precision": 0.62},
                "6": {"recall": 0.55, "precision": 0.60},
            },
            "AutoML": {
                "1": {"recall": 0.85, "precision": 0.82},
                "2": {"recall": 0.82, "precision": 0.80},
                "3": {"recall": 0.78, "precision": 0.76},
                "4": {"recall": 0.72, "precision": 0.70},
                "5": {"recall": 0.60, "precision": 0.58},
                "6": {"recall": 0.50, "precision": 0.55},
            },
        }
        table = compare_model_fairness(models)
        assert "EfficientNetV2" in table
        assert "AutoML" in table
        assert "gap" in table["EfficientNetV2"]
