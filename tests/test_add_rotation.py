"""Tests for scripts/add_rotation.py."""
import numpy as np
import pytest
from pathlib import Path
from PIL import Image

from scripts.add_rotation import (
    rotate_image,
    process_images,
    verify_download,
    verify_rotation,
)


def _make_test_image(path: Path, size: tuple = (64, 64)) -> Path:
    """Create a small solid-color test image."""
    arr = np.full((*size, 3), 128, dtype=np.uint8)
    Image.fromarray(arr).save(str(path))
    return path


# ---------------------------------------------------------------------------
# TestRotateImage
# ---------------------------------------------------------------------------


class TestRotateImage:
    def test_shape_preserved(self):
        img = np.zeros((32, 32, 3), dtype=np.uint8)
        result = rotate_image(img, angle=45.0)
        assert result.shape == img.shape

    def test_dtype_uint8(self):
        img = np.zeros((32, 32, 3), dtype=np.uint8)
        result = rotate_image(img, angle=45.0)
        assert result.dtype == np.uint8

    def test_pixels_change(self):
        img = np.full((64, 64, 3), 128, dtype=np.uint8)
        result = rotate_image(img, angle=45.0)
        assert not np.array_equal(img, result)

    def test_seed_reproducibility(self):
        img = np.full((32, 32, 3), 128, dtype=np.uint8)
        r1 = rotate_image(img, angle=90.0)
        r2 = rotate_image(img, angle=90.0)
        np.testing.assert_array_equal(r1, r2)


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
        assert "top_rotated" in stems
        assert "nested_rotated" in stems

    def test_skips_rotated(self, tmp_path):
        in_dir = tmp_path / "input"
        in_dir.mkdir()
        _make_test_image(in_dir / "img.png")
        _make_test_image(in_dir / "img_rotated.png")

        out_dir = tmp_path / "output"
        count, paths = process_images(str(in_dir), str(out_dir), seed=42)

        assert count == 1
        assert paths[0].stem == "img_rotated"

    def test_returns_correct_paths(self, tmp_path):
        in_dir = tmp_path / "input"
        in_dir.mkdir()
        _make_test_image(in_dir / "test.png")

        out_dir = tmp_path / "output"
        count, paths = process_images(str(in_dir), str(out_dir), seed=42)

        assert count == 1
        assert paths[0].name == "test_rotated.png"
        assert paths[0].parent == out_dir


# ---------------------------------------------------------------------------
# TestVerifyRotation
# ---------------------------------------------------------------------------


class TestVerifyRotation:
    def test_passes_on_valid_rotation(self, tmp_path):
        orig_dir = tmp_path / "orig"
        rotated_dir = tmp_path / "rotated"
        orig_dir.mkdir()

        _make_test_image(orig_dir / "a.png")
        process_images(str(orig_dir), str(rotated_dir), seed=42)

        result = verify_rotation(str(orig_dir), str(rotated_dir))
        assert result["passed"] is True
        assert len(result["missing"]) == 0

    def test_fails_when_images_identical(self, tmp_path):
        orig_dir = tmp_path / "orig"
        rotated_dir = tmp_path / "rotated"
        orig_dir.mkdir()
        rotated_dir.mkdir()

        _make_test_image(orig_dir / "a.png")
        # Copy original as "rotated" (identical pixels)
        _make_test_image(rotated_dir / "a_rotated.png")

        result = verify_rotation(str(orig_dir), str(rotated_dir))
        assert result["passed"] is False


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
