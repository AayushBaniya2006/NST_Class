"""Tests for src.data.transforms — image augmentation pipelines."""
import numpy as np
import pytest
import torch
from PIL import Image

from src.data.transforms import (
    get_train_transforms,
    get_eval_transforms,
    GaussianNoise,
    AUGMENTATION_BUCKETS,
)


def _make_test_image(size=100):
    return Image.fromarray(np.random.randint(0, 255, (size, size, 3), dtype=np.uint8))


class TestTransforms:
    def test_train_transform_output_shape(self):
        tensor = get_train_transforms(224)(_make_test_image())
        assert tensor.shape == (3, 224, 224)

    def test_eval_transform_output_shape(self):
        tensor = get_eval_transforms(224)(_make_test_image())
        assert tensor.shape == (3, 224, 224)

    def test_train_transform_custom_size(self):
        tensor = get_train_transforms(64)(_make_test_image())
        assert tensor.shape == (3, 64, 64)


class TestAugmentationBuckets:
    """Each bucket should produce the correct output shape."""

    @pytest.mark.parametrize("bucket", AUGMENTATION_BUCKETS)
    def test_bucket_output_shape(self, bucket):
        transform = get_train_transforms(224, augmentation=bucket)
        tensor = transform(_make_test_image())
        assert tensor.shape == (3, 224, 224)

    @pytest.mark.parametrize("bucket", AUGMENTATION_BUCKETS)
    def test_bucket_custom_size(self, bucket):
        transform = get_train_transforms(64, augmentation=bucket)
        tensor = transform(_make_test_image())
        assert tensor.shape == (3, 64, 64)

    def test_invalid_bucket_raises(self):
        with pytest.raises(ValueError, match="Unknown augmentation bucket"):
            get_train_transforms(224, augmentation="invalid_bucket")

    def test_control_matches_eval(self):
        """Control bucket should produce same output as eval transforms (no randomness)."""
        img = _make_test_image()
        control = get_train_transforms(224, augmentation="control")(img)
        eval_out = get_eval_transforms(224)(img)
        assert torch.allclose(control, eval_out)


class TestGaussianNoise:
    def test_noise_modifies_tensor(self):
        """With p=1.0, output should differ from input."""
        tensor = torch.rand(3, 64, 64)
        noisy = GaussianNoise(p=1.0, std=0.1)(tensor)
        assert not torch.equal(tensor, noisy)

    def test_noise_preserves_range(self):
        """Output should stay in [0, 1]."""
        tensor = torch.rand(3, 64, 64)
        noisy = GaussianNoise(p=1.0, std=0.5)(tensor)
        assert noisy.min() >= 0.0
        assert noisy.max() <= 1.0

    def test_noise_p_zero_is_identity(self):
        """With p=0.0, output should equal input."""
        tensor = torch.rand(3, 64, 64)
        result = GaussianNoise(p=0.0)(tensor)
        assert torch.equal(tensor, result)

    def test_noise_shape_preserved(self):
        tensor = torch.rand(3, 224, 224)
        noisy = GaussianNoise(p=1.0)(tensor)
        assert noisy.shape == tensor.shape
