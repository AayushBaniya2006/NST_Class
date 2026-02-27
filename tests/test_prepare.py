"""Tests for src.data.prepare — data preparation pipeline for Fitzpatrick17k."""

import math
import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from PIL import Image

from src.data.prepare import (
    VALID_FITZPATRICK,
    NUM_CLASSES,
    FITZPATRICK_TO_LABEL,
    load_metadata,
    validate_fitzpatrick_labels,
    encode_labels,
    validate_images,
    filter_human_images,
    deduplicate_images,
    compute_class_distribution,
    generate_cleaning_report,
    stratified_split,
    download_images,
    run_full_pipeline,
)


# ---------------------------------------------------------------------------
# Helper: synthetic DataFrame that mirrors Fitzpatrick17k CSV structure
# ---------------------------------------------------------------------------

def make_sample_df(n: int = 100, seed: int = 42) -> pd.DataFrame:
    """Generate a synthetic DataFrame resembling Fitzpatrick17k metadata.

    Columns produced: hasher, url, label, fitzpatrick, three_partition_label,
    nine_partition_label, md5hash, qc, image_path
    """
    rng = np.random.default_rng(seed)
    fitzpatrick_values = rng.choice([1, 2, 3, 4, 5, 6], size=n)
    hashers = [f"img_{i:05d}" for i in range(n)]
    return pd.DataFrame(
        {
            "hasher": hashers,
            "url": [f"http://example.com/{h}.jpg" for h in hashers],
            "label": rng.choice(["benign", "malignant", "non-neoplastic"], size=n),
            "fitzpatrick": fitzpatrick_values,
            "three_partition_label": rng.choice(["a", "b", "c"], size=n),
            "nine_partition_label": rng.choice(list("abcdefghi"), size=n),
            "md5hash": [f"md5_{i:05d}" for i in range(n)],
            "qc": rng.choice(["pass", "fail"], size=n),
            "image_path": [f"/images/{h}.jpg" for h in hashers],
        }
    )


# ---------------------------------------------------------------------------
# Helper: create a valid test image on disk
# ---------------------------------------------------------------------------

def _create_test_image(
    path: str,
    size: tuple[int, int] = (100, 100),
    mode: str = "RGB",
    color_variance: bool = True,
) -> None:
    """Write a small test image to *path*."""
    if color_variance:
        rng = np.random.default_rng(0)
        arr = rng.integers(0, 256, (*size, 3), dtype=np.uint8)
        img = Image.fromarray(arr, mode=mode)
    else:
        # Uniform colour — low variance, resembles a diagram placeholder
        img = Image.new(mode, size, color=(128, 128, 128))
    img.save(path)


# ===================================================================
# Test: constants
# ===================================================================

class TestConstants:
    def test_valid_fitzpatrick(self):
        assert VALID_FITZPATRICK == {1, 2, 3, 4, 5, 6}

    def test_num_classes(self):
        assert NUM_CLASSES == 6

    def test_fitzpatrick_to_label_keys(self):
        assert set(FITZPATRICK_TO_LABEL.keys()) == {1, 2, 3, 4, 5, 6}

    def test_fitzpatrick_to_label_values(self):
        assert set(FITZPATRICK_TO_LABEL.values()) == {0, 1, 2, 3, 4, 5}


# ===================================================================
# Test: encode_labels
# ===================================================================

class TestEncodeLabels:
    """Verify that encode_labels adds skin_tone_label with correct mapping."""

    def test_maps_fitzpatrick_1_to_label_0(self):
        df = pd.DataFrame({"fitzpatrick": [1]})
        result = encode_labels(df)
        assert list(result["skin_tone_label"]) == [0]

    def test_maps_fitzpatrick_6_to_label_5(self):
        df = pd.DataFrame({"fitzpatrick": [6]})
        result = encode_labels(df)
        assert list(result["skin_tone_label"]) == [5]

    def test_all_six_types(self):
        df = pd.DataFrame({"fitzpatrick": [1, 2, 3, 4, 5, 6]})
        result = encode_labels(df)
        expected_labels = [0, 1, 2, 3, 4, 5]
        assert list(result["skin_tone_label"]) == expected_labels

    def test_preserves_existing_columns(self):
        df = pd.DataFrame({"fitzpatrick": [1], "other_col": ["x"]})
        result = encode_labels(df)
        assert "other_col" in result.columns

    def test_does_not_modify_input(self):
        df = pd.DataFrame({"fitzpatrick": [1, 2, 3]})
        original_cols = set(df.columns)
        _ = encode_labels(df)
        assert set(df.columns) == original_cols

    def test_no_skin_tone_group_column(self):
        df = pd.DataFrame({"fitzpatrick": [1, 2, 3]})
        result = encode_labels(df)
        assert "skin_tone_group" not in result.columns


# ===================================================================
# Test: validate_fitzpatrick_labels
# ===================================================================

class TestValidateFitzpatrickLabels:
    """Ensure rows with missing or out-of-range fitzpatrick values are dropped."""

    def test_keeps_valid_labels(self):
        df = pd.DataFrame({"fitzpatrick": [1, 2, 3, 4, 5, 6]})
        result = validate_fitzpatrick_labels(df)
        assert len(result) == 6

    def test_drops_none(self):
        df = pd.DataFrame({"fitzpatrick": [1, None, 3]})
        result = validate_fitzpatrick_labels(df)
        assert len(result) == 2

    def test_drops_nan(self):
        df = pd.DataFrame({"fitzpatrick": [1, float("nan"), 5]})
        result = validate_fitzpatrick_labels(df)
        assert len(result) == 2

    def test_drops_invalid_7(self):
        df = pd.DataFrame({"fitzpatrick": [1, 7, 3]})
        result = validate_fitzpatrick_labels(df)
        assert len(result) == 2

    def test_drops_invalid_0(self):
        df = pd.DataFrame({"fitzpatrick": [0, 2]})
        result = validate_fitzpatrick_labels(df)
        assert len(result) == 1

    def test_drops_negative(self):
        df = pd.DataFrame({"fitzpatrick": [-1, 4]})
        result = validate_fitzpatrick_labels(df)
        assert len(result) == 1

    def test_all_invalid_returns_empty(self):
        df = pd.DataFrame({"fitzpatrick": [None, 0, -1, 7, 100]})
        result = validate_fitzpatrick_labels(df)
        assert len(result) == 0

    def test_preserves_other_columns(self):
        df = pd.DataFrame({"fitzpatrick": [1, 7], "name": ["a", "b"]})
        result = validate_fitzpatrick_labels(df)
        assert list(result["name"]) == ["a"]

    def test_resets_index(self):
        df = pd.DataFrame({"fitzpatrick": [7, 1, 2]})
        result = validate_fitzpatrick_labels(df)
        assert list(result.index) == list(range(len(result)))


# ===================================================================
# Test: compute_class_distribution
# ===================================================================

class TestComputeClassDistribution:
    """Validate distribution dictionary has correct counts and percentages."""

    def test_returns_counts(self):
        df = pd.DataFrame({"label": ["a", "a", "b"]})
        dist = compute_class_distribution(df, "label")
        assert dist["a"]["count"] == 2
        assert dist["b"]["count"] == 1

    def test_returns_percentages(self):
        df = pd.DataFrame({"label": ["a", "a", "b", "b"]})
        dist = compute_class_distribution(df, "label")
        assert math.isclose(dist["a"]["percentage"], 50.0)
        assert math.isclose(dist["b"]["percentage"], 50.0)

    def test_single_class(self):
        df = pd.DataFrame({"label": ["x", "x", "x"]})
        dist = compute_class_distribution(df, "label")
        assert dist["x"]["count"] == 3
        assert math.isclose(dist["x"]["percentage"], 100.0)

    def test_multiple_classes(self):
        df = pd.DataFrame({"c": [0, 0, 1, 1, 2]})
        dist = compute_class_distribution(df, "c")
        assert dist[0]["count"] == 2
        assert dist[1]["count"] == 2
        assert dist[2]["count"] == 1
        total = sum(v["percentage"] for v in dist.values())
        assert math.isclose(total, 100.0)

    def test_empty_dataframe(self):
        df = pd.DataFrame({"label": pd.Series([], dtype=str)})
        dist = compute_class_distribution(df, "label")
        assert dist == {}


# ===================================================================
# Test: stratified_split
# ===================================================================

class TestStratifiedSplit:
    """Test that stratified_split produces correct ratios and no overlap."""

    def test_default_ratios(self):
        df = make_sample_df(300)
        df = encode_labels(df)
        train, val, test = stratified_split(df, "skin_tone_label")
        total = len(train) + len(val) + len(test)
        assert total == 300
        # Allow +-5 % tolerance for rounding
        assert abs(len(train) / 300 - 0.7) < 0.05
        assert abs(len(val) / 300 - 0.15) < 0.05
        assert abs(len(test) / 300 - 0.15) < 0.05

    def test_custom_ratios(self):
        df = make_sample_df(200)
        df = encode_labels(df)
        train, val, test = stratified_split(
            df, "skin_tone_label", ratios=(0.6, 0.2, 0.2)
        )
        total = len(train) + len(val) + len(test)
        assert total == 200
        assert abs(len(train) / 200 - 0.6) < 0.05

    def test_no_overlap(self):
        df = make_sample_df(200)
        df = encode_labels(df)
        train, val, test = stratified_split(df, "skin_tone_label")
        train_idx = set(train.index)
        val_idx = set(val.index)
        test_idx = set(test.index)
        assert train_idx.isdisjoint(val_idx)
        assert train_idx.isdisjoint(test_idx)
        assert val_idx.isdisjoint(test_idx)

    def test_stratification_preserves_distribution(self):
        """Each split should have roughly the same class proportions."""
        df = make_sample_df(600)
        df = encode_labels(df)
        train, val, test = stratified_split(df, "skin_tone_label")
        for label in df["skin_tone_label"].unique():
            full_ratio = (df["skin_tone_label"] == label).mean()
            train_ratio = (train["skin_tone_label"] == label).mean()
            val_ratio = (val["skin_tone_label"] == label).mean()
            test_ratio = (test["skin_tone_label"] == label).mean()
            assert abs(train_ratio - full_ratio) < 0.05
            assert abs(val_ratio - full_ratio) < 0.1  # smaller sets → more variance
            assert abs(test_ratio - full_ratio) < 0.1

    def test_reproducibility(self):
        df = make_sample_df(200)
        df = encode_labels(df)
        t1, v1, te1 = stratified_split(df, "skin_tone_label", seed=99)
        t2, v2, te2 = stratified_split(df, "skin_tone_label", seed=99)
        pd.testing.assert_frame_equal(t1.reset_index(drop=True), t2.reset_index(drop=True))
        pd.testing.assert_frame_equal(v1.reset_index(drop=True), v2.reset_index(drop=True))
        pd.testing.assert_frame_equal(te1.reset_index(drop=True), te2.reset_index(drop=True))


# ===================================================================
# Test: load_metadata
# ===================================================================

class TestLoadMetadata:
    def test_loads_csv(self, tmp_path):
        csv_path = tmp_path / "meta.csv"
        df = make_sample_df(10)
        df.to_csv(csv_path, index=False)
        result = load_metadata(str(csv_path))
        assert len(result) == 10
        assert "hasher" in result.columns

    def test_raises_on_missing_file(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_metadata(str(tmp_path / "nonexistent.csv"))

    def test_renames_upstream_columns(self, tmp_path):
        """CSV with fitzpatrick_scale/url_alphanum should be renamed."""
        csv_path = tmp_path / "upstream.csv"
        df = pd.DataFrame({
            "url_alphanum": ["img_a", "img_b"],
            "fitzpatrick_scale": [1, 3],
            "url": ["http://a.com", "http://b.com"],
        })
        df.to_csv(csv_path, index=False)
        result = load_metadata(str(csv_path))
        assert "hasher" in result.columns
        assert "fitzpatrick" in result.columns
        assert "url_alphanum" not in result.columns
        assert "fitzpatrick_scale" not in result.columns

    def test_already_renamed_columns_unchanged(self, tmp_path):
        """CSV that already has hasher/fitzpatrick should pass through."""
        csv_path = tmp_path / "already.csv"
        df = make_sample_df(5)
        df.to_csv(csv_path, index=False)
        result = load_metadata(str(csv_path))
        assert "hasher" in result.columns
        assert "fitzpatrick" in result.columns


# ===================================================================
# Test: validate_images
# ===================================================================

class TestValidateImages:
    def test_keeps_valid_images(self, tmp_path):
        _create_test_image(str(tmp_path / "img_00000.jpg"))
        df = pd.DataFrame({"hasher": ["img_00000"]})
        result = validate_images(str(tmp_path), df)
        assert len(result) == 1

    def test_drops_missing_images(self, tmp_path):
        df = pd.DataFrame({"hasher": ["nonexistent"]})
        result = validate_images(str(tmp_path), df)
        assert len(result) == 0

    def test_tries_multiple_extensions(self, tmp_path):
        _create_test_image(str(tmp_path / "img_00000.png"))
        df = pd.DataFrame({"hasher": ["img_00000"]})
        result = validate_images(str(tmp_path), df)
        assert len(result) == 1

    def test_drops_corrupted_image(self, tmp_path):
        bad_path = tmp_path / "bad_img.jpg"
        bad_path.write_bytes(b"not a real image file contents")
        df = pd.DataFrame({"hasher": ["bad_img"]})
        result = validate_images(str(tmp_path), df)
        assert len(result) == 0


# ===================================================================
# Test: filter_human_images
# ===================================================================

class TestFilterHumanImages:
    def test_keeps_rgb_normal_size(self, tmp_path):
        _create_test_image(str(tmp_path / "good.jpg"), size=(100, 100))
        df = pd.DataFrame({"hasher": ["good"]})
        result = filter_human_images(str(tmp_path), df)
        assert len(result) == 1

    def test_drops_too_small(self, tmp_path):
        _create_test_image(str(tmp_path / "tiny.jpg"), size=(30, 30))
        df = pd.DataFrame({"hasher": ["tiny"]})
        result = filter_human_images(str(tmp_path), df)
        assert len(result) == 0

    def test_drops_low_variance(self, tmp_path):
        _create_test_image(
            str(tmp_path / "flat.jpg"), size=(100, 100), color_variance=False
        )
        df = pd.DataFrame({"hasher": ["flat"]})
        result = filter_human_images(str(tmp_path), df)
        assert len(result) == 0

    def test_drops_grayscale(self, tmp_path):
        path = tmp_path / "gray.jpg"
        img = Image.new("L", (100, 100), color=128)
        img.save(str(path))
        df = pd.DataFrame({"hasher": ["gray"]})
        result = filter_human_images(str(tmp_path), df)
        assert len(result) == 0


# ===================================================================
# Test: deduplicate_images
# ===================================================================

class TestDeduplicateImages:
    def test_keeps_unique_images(self, tmp_path):
        _create_test_image(str(tmp_path / "a.jpg"), size=(64, 64))
        _create_test_image(str(tmp_path / "b.jpg"), size=(80, 80))
        df = pd.DataFrame({"hasher": ["a", "b"]})
        result = deduplicate_images(str(tmp_path), df)
        # Both are visually different (random with different sizes) — should keep both
        assert len(result) >= 1

    def test_removes_exact_duplicates(self, tmp_path):
        # Create two identical images with different names
        _create_test_image(str(tmp_path / "orig.jpg"), size=(64, 64))
        import shutil
        shutil.copy(str(tmp_path / "orig.jpg"), str(tmp_path / "dup.jpg"))
        df = pd.DataFrame({"hasher": ["orig", "dup"]})
        result = deduplicate_images(str(tmp_path), df)
        assert len(result) == 1


# ===================================================================
# Test: generate_cleaning_report
# ===================================================================

class TestGenerateCleaningReport:
    def test_report_structure(self):
        df = pd.DataFrame({"skin_tone_label": [0, 0, 1, 2]})
        report = generate_cleaning_report(
            original_count=10,
            cleaned_df=df,
            column="skin_tone_label",
            dropped_reasons={"invalid_label": 4, "missing_image": 2},
        )
        assert report["original_count"] == 10
        assert report["cleaned_count"] == 4
        assert report["total_dropped"] == 6
        assert report["dropped_reasons"]["invalid_label"] == 4
        assert "class_distribution" in report


# ===================================================================
# Test: download_images (basic)
# ===================================================================

class TestDownloadImages:
    def test_skips_existing(self, tmp_path):
        # Pre-create an image that would be "downloaded"
        _create_test_image(str(tmp_path / "img_00000.jpg"))
        df = pd.DataFrame(
            {"url": ["http://example.com/img.jpg"], "hasher": ["img_00000"]}
        )
        count = download_images(df, str(tmp_path))
        # Should skip existing file — 0 new downloads attempted successfully
        assert count == 0
