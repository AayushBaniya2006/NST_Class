"""Tests for scripts/add_gaussian_noise.py."""
import numpy as np
import pytest
from pathlib import Path
from PIL import Image

from scripts.add_gaussian_noise import (
    add_gaussian_noise,
    process_images,
    verify_download,
    verify_noise,
)


def _make_test_image(path: Path, size: tuple = (64, 64)) -> Path:
    """Create a small solid-color test image."""
    arr = np.full((*size, 3), 128, dtype=np.uint8)
    Image.fromarray(arr).save(str(path))
    return path


# ---------------------------------------------------------------------------
# TestAddGaussianNoise
# ---------------------------------------------------------------------------


class TestAddGaussianNoise:
    def test_shape_preserved(self):
        img = np.zeros((32, 32, 3), dtype=np.uint8)
        result = add_gaussian_noise(img, rng=np.random.default_rng(0))
        assert result.shape == img.shape

    def test_dtype_uint8(self):
        img = np.zeros((32, 32, 3), dtype=np.uint8)
        result = add_gaussian_noise(img, rng=np.random.default_rng(0))
        assert result.dtype == np.uint8

    def test_clipped_0_255(self):
        img = np.full((32, 32, 3), 250, dtype=np.uint8)
        result = add_gaussian_noise(img, std=100.0, rng=np.random.default_rng(0))
        assert result.min() >= 0
        assert result.max() <= 255

    def test_pixels_change(self):
        img = np.full((64, 64, 3), 128, dtype=np.uint8)
        result = add_gaussian_noise(img, std=25.0, rng=np.random.default_rng(0))
        assert not np.array_equal(img, result)

    def test_seed_reproducibility(self):
        img = np.full((32, 32, 3), 128, dtype=np.uint8)
        r1 = add_gaussian_noise(img, rng=np.random.default_rng(42))
        r2 = add_gaussian_noise(img, rng=np.random.default_rng(42))
        np.testing.assert_array_equal(r1, r2)

    def test_different_seeds_differ(self):
        img = np.full((32, 32, 3), 128, dtype=np.uint8)
        r1 = add_gaussian_noise(img, rng=np.random.default_rng(1))
        r2 = add_gaussian_noise(img, rng=np.random.default_rng(2))
        assert not np.array_equal(r1, r2)


# ---------------------------------------------------------------------------
# TestProcessImages
# ---------------------------------------------------------------------------


class TestProcessImages:
    def test_processes_all_images(self, tmp_path):
        in_dir = tmp_path / "input"
        in_dir.mkdir()
        for i in range(3):
            _make_test_image(in_dir / f"img{i}.png")

        out_dir = tmp_path / "output"
        count, paths = process_images(str(in_dir), str(out_dir), seed=42)

        assert count == 3
        assert len(paths) == 3
        assert all(p.exists() for p in paths)

    def test_recursive_finds_nested(self, tmp_path):
        in_dir = tmp_path / "input"
        sub = in_dir / "subdir"
        sub.mkdir(parents=True)
        _make_test_image(in_dir / "top.png")
        _make_test_image(sub / "nested.jpg")

        out_dir = tmp_path / "output"
        count, paths = process_images(str(in_dir), str(out_dir), seed=42)

        assert count == 2
        stems = {p.stem for p in paths}
        assert "top_noised" in stems
        assert "nested_noised" in stems

    def test_skips_noised(self, tmp_path):
        in_dir = tmp_path / "input"
        in_dir.mkdir()
        _make_test_image(in_dir / "img.png")
        _make_test_image(in_dir / "img_noised.png")

        out_dir = tmp_path / "output"
        count, paths = process_images(str(in_dir), str(out_dir), seed=42)

        assert count == 1
        assert paths[0].stem == "img_noised"

    def test_seed_reproducibility(self, tmp_path):
        in_dir = tmp_path / "input"
        in_dir.mkdir()
        _make_test_image(in_dir / "img.png")

        out1 = tmp_path / "out1"
        out2 = tmp_path / "out2"
        process_images(str(in_dir), str(out1), seed=42)
        process_images(str(in_dir), str(out2), seed=42)

        img1 = np.array(Image.open(out1 / "img_noised.png"))
        img2 = np.array(Image.open(out2 / "img_noised.png"))
        np.testing.assert_array_equal(img1, img2)

    def test_returns_correct_paths(self, tmp_path):
        in_dir = tmp_path / "input"
        in_dir.mkdir()
        _make_test_image(in_dir / "test.png")

        out_dir = tmp_path / "output"
        count, paths = process_images(str(in_dir), str(out_dir), seed=42)

        assert count == 1
        assert paths[0].name == "test_noised.png"
        assert paths[0].parent == out_dir


# ---------------------------------------------------------------------------
# TestVerifyNoise
# ---------------------------------------------------------------------------


class TestVerifyNoise:
    def test_passes_on_valid_noise(self, tmp_path):
        orig_dir = tmp_path / "orig"
        noised_dir = tmp_path / "noised"
        orig_dir.mkdir()

        _make_test_image(orig_dir / "a.png")
        process_images(str(orig_dir), str(noised_dir), seed=42)

        result = verify_noise(str(orig_dir), str(noised_dir))
        assert result["passed"] is True
        assert len(result["missing"]) == 0

    def test_fails_when_images_identical(self, tmp_path):
        orig_dir = tmp_path / "orig"
        noised_dir = tmp_path / "noised"
        orig_dir.mkdir()
        noised_dir.mkdir()

        _make_test_image(orig_dir / "a.png")
        # Copy original as "noised" (identical pixels)
        _make_test_image(noised_dir / "a_noised.png")

        result = verify_noise(str(orig_dir), str(noised_dir))
        assert result["passed"] is False

    def test_noise_stats_in_range(self, tmp_path):
        orig_dir = tmp_path / "orig"
        noised_dir = tmp_path / "noised"
        orig_dir.mkdir()

        # Use multiple images for better stats
        for i in range(10):
            _make_test_image(orig_dir / f"img{i}.png")

        process_images(str(orig_dir), str(noised_dir), std=25.0, seed=42)

        result = verify_noise(str(orig_dir), str(noised_dir), expected_std=25.0)
        assert result["passed"] is True
        assert abs(result["mean_noise_mean"]) < 12.5  # within 0.5 * std
        assert abs(result["mean_noise_std"] - 25.0) < 12.5


# ---------------------------------------------------------------------------
# TestVerifyDownload
# ---------------------------------------------------------------------------


class TestVerifyDownload:
    def test_counts_valid_images(self, tmp_path):
        for i in range(5):
            _make_test_image(tmp_path / f"img{i}.png")

        result = verify_download(str(tmp_path))
        assert result["total"] == 5
        assert result["valid"] == 5
        assert result["corrupt"] == 0

    def test_detects_corrupt(self, tmp_path):
        _make_test_image(tmp_path / "good.png")
        # Write garbage bytes as a "corrupt" image
        corrupt = tmp_path / "bad.jpg"
        corrupt.write_bytes(b"not an image at all")

        result = verify_download(str(tmp_path))
        assert result["total"] == 2
        assert result["valid"] == 1
        assert result["corrupt"] == 1
