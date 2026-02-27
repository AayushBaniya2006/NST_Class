import pytest
import numpy as np
from src.training.trainer import compute_class_weights, EarlyStopping


class TestComputeClassWeights:
    def test_balanced_classes(self):
        labels = [0] * 100 + [1] * 100 + [2] * 100
        weights = compute_class_weights(labels, num_classes=3)
        assert len(weights) == 3
        assert abs(weights[0] - weights[1]) < 0.1

    def test_imbalanced_classes(self):
        labels = [0] * 500 + [1] * 100 + [2] * 50
        weights = compute_class_weights(labels, num_classes=3)
        assert weights[2] > weights[1] > weights[0]


class TestEarlyStopping:
    def test_no_stop_when_improving(self):
        es = EarlyStopping(patience=3)
        assert not es.step(1.0)
        assert not es.step(0.9)
        assert not es.step(0.8)

    def test_stops_after_patience(self):
        es = EarlyStopping(patience=3)
        es.step(0.5)
        es.step(0.6)
        es.step(0.7)
        assert es.step(0.8)

    def test_resets_on_improvement(self):
        es = EarlyStopping(patience=3)
        es.step(0.5)
        es.step(0.6)
        es.step(0.4)  # improvement resets counter
        es.step(0.5)
        assert not es.step(0.6)  # only 2 non-improvements since reset
