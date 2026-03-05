"""Tests for src.data.transforms — image augmentation pipelines."""
import numpy as np
from PIL import Image

from src.data.transforms import get_train_transforms, get_eval_transforms


class TestTransforms:
    def test_train_transform_output_shape(self):
        img = Image.fromarray(np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8))
        transform = get_train_transforms(224)
        tensor = transform(img)
        assert tensor.shape == (3, 224, 224)

    def test_eval_transform_output_shape(self):
        img = Image.fromarray(np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8))
        transform = get_eval_transforms(224)
        tensor = transform(img)
        assert tensor.shape == (3, 224, 224)

    def test_train_transform_custom_size(self):
        img = Image.fromarray(np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8))
        transform = get_train_transforms(64)
        tensor = transform(img)
        assert tensor.shape == (3, 64, 64)
