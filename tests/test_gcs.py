"""Tests for src/utils/gcs.py."""
import pandas as pd
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock

from src.utils.gcs import (
    _find_image_extension,
    generate_automl_csv,
)


# ---------------------------------------------------------------------------
# TestFindImageExtension
# ---------------------------------------------------------------------------


class TestFindImageExtension:
    def test_finds_jpg(self, tmp_path):
        (tmp_path / "abc.jpg").touch()
        assert _find_image_extension(str(tmp_path), "abc") == ".jpg"

    def test_finds_png(self, tmp_path):
        (tmp_path / "abc.png").touch()
        assert _find_image_extension(str(tmp_path), "abc") == ".png"

    def test_defaults_to_jpg_when_missing(self, tmp_path):
        assert _find_image_extension(str(tmp_path), "nonexistent") == ".jpg"

    def test_prefers_jpg_over_png(self, tmp_path):
        (tmp_path / "abc.jpg").touch()
        (tmp_path / "abc.png").touch()
        # _IMAGE_EXTENSIONS has .jpg before .png
        assert _find_image_extension(str(tmp_path), "abc") == ".jpg"


# ---------------------------------------------------------------------------
# TestGenerateAutomlCsv
# ---------------------------------------------------------------------------


class TestGenerateAutomlCsv:
    def _make_df(self):
        return pd.DataFrame(
            {
                "hasher": ["aaa", "bbb", "ccc"],
                "skin_tone_label": [0, 1, 2],
                "split": ["train", "val", "test"],
            }
        )

    def test_generates_correct_format(self, tmp_path):
        df = self._make_df()
        out = str(tmp_path / "manifest.csv")
        generate_automl_csv(df, "gs://bucket/images", output_path=out)

        lines = Path(out).read_text().strip().split("\n")
        assert len(lines) == 3
        # Each line: ML_USE,gcs_path,label
        parts = lines[0].split(",")
        assert len(parts) == 3

    def test_split_mapping(self, tmp_path):
        df = self._make_df()
        out = str(tmp_path / "manifest.csv")
        generate_automl_csv(df, "gs://bucket/images", output_path=out)

        lines = Path(out).read_text().strip().split("\n")
        ml_uses = [l.split(",")[0] for l in lines]
        assert "TRAINING" in ml_uses
        assert "VALIDATION" in ml_uses
        assert "TEST" in ml_uses

    def test_uses_image_dir_for_extensions(self, tmp_path):
        img_dir = tmp_path / "images"
        img_dir.mkdir()
        (img_dir / "aaa.png").touch()
        (img_dir / "bbb.jpg").touch()
        (img_dir / "ccc.jpeg").touch()

        df = self._make_df()
        out = str(tmp_path / "manifest.csv")
        generate_automl_csv(
            df, "gs://bucket/images", output_path=out, image_dir=str(img_dir)
        )

        lines = Path(out).read_text().strip().split("\n")
        paths = [l.split(",")[1] for l in lines]
        assert any(".png" in p for p in paths)
        assert any(".jpg" in p for p in paths)

    def test_defaults_to_jpg_without_image_dir(self, tmp_path):
        df = self._make_df()
        out = str(tmp_path / "manifest.csv")
        generate_automl_csv(df, "gs://bucket/images", output_path=out)

        lines = Path(out).read_text().strip().split("\n")
        for line in lines:
            gcs_path = line.split(",")[1]
            assert gcs_path.endswith(".jpg")


# ---------------------------------------------------------------------------
# TestUploadDirectoryToGcs (mocked)
# ---------------------------------------------------------------------------


class TestUploadDirectoryToGcs:
    def test_upload_counts_files(self, tmp_path):
        """Verify upload_directory_to_gcs uploads all files recursively."""
        (tmp_path / "a.jpg").touch()
        (tmp_path / "b.jpg").touch()
        sub = tmp_path / "sub"
        sub.mkdir()
        (sub / "c.png").touch()

        mock_storage_mod = MagicMock()
        import sys
        orig = {k: sys.modules.get(k) for k in ["google", "google.cloud", "google.cloud.storage"]}
        sys.modules["google"] = MagicMock()
        sys.modules["google.cloud"] = MagicMock()
        sys.modules["google.cloud.storage"] = mock_storage_mod

        try:
            from src.utils.gcs import upload_directory_to_gcs
            count = upload_directory_to_gcs(str(tmp_path), "bucket", "pfx")
            assert count == 3
        finally:
            for k, v in orig.items():
                if v is None:
                    sys.modules.pop(k, None)
                else:
                    sys.modules[k] = v


class TestUploadFileToGcs:
    def test_returns_gcs_uri(self, tmp_path):
        test_file = tmp_path / "test.csv"
        test_file.write_text("data")

        mock_storage_mod = MagicMock()
        import sys
        orig = {k: sys.modules.get(k) for k in ["google", "google.cloud", "google.cloud.storage"]}
        sys.modules["google"] = MagicMock()
        sys.modules["google.cloud"] = MagicMock()
        sys.modules["google.cloud.storage"] = mock_storage_mod

        try:
            from src.utils.gcs import upload_file_to_gcs
            uri = upload_file_to_gcs(str(test_file), "my-bucket", "path/test.csv")
            assert uri == "gs://my-bucket/path/test.csv"
        finally:
            for k, v in orig.items():
                if v is None:
                    sys.modules.pop(k, None)
                else:
                    sys.modules[k] = v
