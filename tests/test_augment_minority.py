"""Tests for scripts/augment_minority.py."""
import numpy as np
import pytest
from pathlib import Path
from PIL import Image

from scripts.augment_minority import (
    get_augmentation_pipeline,
    augment_images,
    verify_augmentation,
)


def _make_test_image(path: Path, size: tuple = (64, 64)) -> Path:
    """Create a simple test image with varied pixels."""
    rng = np.random.default_rng(42)
    arr = rng.integers(50, 200, (*size, 3), dtype=np.uint8)
    Image.fromarray(arr).save(str(path))
    return path


# ---------------------------------------------------------------------------
# TestGetAugmentationPipeline
# ---------------------------------------------------------------------------


class TestGetAugmentationPipeline:
    def test_returns_compose(self):
        import albumentations as A

        pipeline = get_augmentation_pipeline()
        assert isinstance(pipeline, A.Compose)

    def test_processes_default_size(self):
        pipeline = get_augmentation_pipeline(image_size=224)
        arr = np.full((224, 224, 3), 128, dtype=np.uint8)
        result = pipeline(image=arr)["image"]
        assert result.shape[0] > 0 and result.shape[1] > 0

    def test_processes_custom_size(self):
        pipeline = get_augmentation_pipeline(image_size=128)
        arr = np.full((128, 128, 3), 128, dtype=np.uint8)
        result = pipeline(image=arr)["image"]
        assert result.shape[0] > 0 and result.shape[1] > 0

    def test_no_forbidden_transforms(self):
        """Verify no brightness, contrast, color jitter, or blur transforms."""
        import albumentations as A

        pipeline = get_augmentation_pipeline()
        forbidden_types = (
            A.RandomBrightnessContrast,
            A.ColorJitter,
            A.Blur,
            A.GaussianBlur,
            A.MotionBlur,
            A.MedianBlur,
        )
        for t in pipeline.transforms:
            assert not isinstance(t, forbidden_types), (
                f"Forbidden transform found: {type(t).__name__}"
            )


# ---------------------------------------------------------------------------
# TestAugmentImages
# ---------------------------------------------------------------------------


class TestAugmentImages:
    def test_creates_target_count_images(self, tmp_path):
        src = tmp_path / "src"
        src.mkdir()
        for i in range(3):
            _make_test_image(src / f"img_{i}.jpg")

        out = tmp_path / "out"
        img_ids = [f"img_{i}.jpg" for i in range(3)]
        created, paths = augment_images(str(src), str(out), img_ids, target_count=10)
        assert created == 10
        assert len(list(out.iterdir())) == 10

    def test_returns_correct_count_and_paths(self, tmp_path):
        src = tmp_path / "src"
        src.mkdir()
        _make_test_image(src / "a.jpg")

        out = tmp_path / "out"
        created, paths = augment_images(str(src), str(out), ["a.jpg"], target_count=5)
        assert created == len(paths)
        assert created == 5

    def test_output_images_are_valid(self, tmp_path):
        src = tmp_path / "src"
        src.mkdir()
        _make_test_image(src / "a.jpg")

        out = tmp_path / "out"
        _, paths = augment_images(str(src), str(out), ["a.jpg"], target_count=3)
        for p in paths:
            img = Image.open(p)
            img.verify()  # raises if corrupt

    def test_seed_produces_consistent_count(self, tmp_path):
        """Same seed produces same number of outputs and same filenames."""
        src = tmp_path / "src"
        src.mkdir()
        _make_test_image(src / "a.jpg")

        out1 = tmp_path / "out1"
        out2 = tmp_path / "out2"
        c1, p1 = augment_images(str(src), str(out1), ["a.jpg"], target_count=3, seed=99)
        c2, p2 = augment_images(str(src), str(out2), ["a.jpg"], target_count=3, seed=99)

        assert c1 == c2
        assert [p.name for p in p1] == [p.name for p in p2]

    def test_different_seeds_differ(self, tmp_path):
        src = tmp_path / "src"
        src.mkdir()
        _make_test_image(src / "a.jpg")

        out1 = tmp_path / "out1"
        out2 = tmp_path / "out2"
        augment_images(str(src), str(out1), ["a.jpg"], target_count=3, seed=1)
        augment_images(str(src), str(out2), ["a.jpg"], target_count=3, seed=999)

        any_differ = False
        for f1, f2 in zip(sorted(out1.iterdir()), sorted(out2.iterdir())):
            arr1 = np.array(Image.open(f1))
            arr2 = np.array(Image.open(f2))
            if not np.array_equal(arr1, arr2):
                any_differ = True
                break
        assert any_differ

    def test_empty_img_ids_returns_zero(self, tmp_path):
        src = tmp_path / "src"
        src.mkdir()
        out = tmp_path / "out"
        created, paths = augment_images(str(src), str(out), [], target_count=5)
        assert created == 0
        assert paths == []

    def test_nonexistent_images_returns_zero(self, tmp_path):
        src = tmp_path / "src"
        src.mkdir()
        out = tmp_path / "out"
        created, paths = augment_images(
            str(src), str(out), ["does_not_exist.jpg"], target_count=5
        )
        assert created == 0

    def test_cycles_through_sources(self, tmp_path):
        src = tmp_path / "src"
        src.mkdir()
        _make_test_image(src / "a.jpg")
        _make_test_image(src / "b.jpg")

        out = tmp_path / "out"
        _, paths = augment_images(
            str(src), str(out), ["a.jpg", "b.jpg"], target_count=6
        )
        # Filenames are like a_combined_0.jpg — check both source stems appear
        stems = [p.stem for p in paths]
        has_a = any("a_" in s for s in stems)
        has_b = any("b_" in s for s in stems)
        assert has_a and has_b


# ---------------------------------------------------------------------------
# TestVerifyAugmentation
# ---------------------------------------------------------------------------


class TestVerifyAugmentation:
    def test_passes_on_valid_augmentation(self, tmp_path):
        src = tmp_path / "src"
        src.mkdir()
        _make_test_image(src / "a.jpg")

        out = tmp_path / "out"
        augment_images(str(src), str(out), ["a.jpg"], target_count=3, seed=42)

        result = verify_augmentation(str(src), str(out))
        assert result["passed"] is True
        assert result["corrupt"] == 0
        assert result["total"] == 3

    def test_fails_on_corrupt_image(self, tmp_path):
        src = tmp_path / "src"
        src.mkdir()
        _make_test_image(src / "a.jpg")

        out = tmp_path / "out"
        out.mkdir()
        # Write garbage bytes as an "augmented" image
        (out / "a_aug_0.jpg").write_bytes(b"not an image")

        result = verify_augmentation(str(src), str(out))
        assert result["corrupt"] > 0

    def test_fails_on_empty_dir(self, tmp_path):
        src = tmp_path / "src"
        src.mkdir()
        out = tmp_path / "out"
        out.mkdir()

        result = verify_augmentation(str(src), str(out))
        assert result["passed"] is False
        assert result["total"] == 0
