"""Tests for scripts/download_all_sources.py."""
import os
import sys
import zipfile as zf
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest
from PIL import Image as PILImage

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts import download_all_sources as dl


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def tmp_image_dir(tmp_path):
    return str(tmp_path / "images")


@pytest.fixture
def fitz_csv(tmp_path):
    """Minimal fitzpatrick17k.csv with 2 Atlas + 1 DermaAmin URL."""
    csv_path = tmp_path / "fitzpatrick17k.csv"
    csv_path.write_text(
        "md5hash,fitzpatrick_scale,fitzpatrick_centaur,label,nine_partition_label,"
        "three_partition_label,qc,url,url_alphanum\n"
        "abc123,3,3,test,,,,http://atlasdermatologico.com.br/img?imageId=1,abc.jpg\n"
        "def456,1,1,test,,,,http://atlasdermatologico.com.br/img?imageId=2,def.jpg\n"
        "ghi789,2,2,test,,,,https://www.dermaamin.com/site/images/test.jpg,ghi.jpg\n"
    )
    return str(csv_path)


@pytest.fixture
def combined_df():
    """combined_dataset.csv rows for the 3 fitz images."""
    return pd.DataFrame({
        "img_id": ["abc123.jpg", "def456.jpg", "ghi789.jpg"],
        "fitzpatrick_scale": [3, 1, 2],
        "source": ["fitzpatrick17k", "fitzpatrick17k", "fitzpatrick17k"],
    })


@pytest.fixture
def scin_combined_df():
    return pd.DataFrame({
        "img_id": ["111.png", "222.png", "333.png"],
        "fitzpatrick_scale": [2, 3, 4],
        "source": ["scin", "scin", "scin"],
    })


@pytest.fixture
def pad_combined_df():
    return pd.DataFrame({
        "img_id": ["PAT_1_1_1.png", "PAT_2_2_2.png"],
        "fitzpatrick_scale": [2, 3],
        "source": ["pad-ufes", "pad-ufes"],
    })


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fake_download_single(args):
    """Write a tiny valid JPEG to the dest path."""
    url, dest = args
    if "dermaamin" in url:
        return False
    # Minimal JPEG: FF D8 FF E0 header
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    with open(dest, "wb") as f:
        f.write(b"\xff\xd8\xff\xe0" + b"\x00" * 100)
    return True


# ---------------------------------------------------------------------------
# Fitzpatrick17k tests
# ---------------------------------------------------------------------------

def test_download_fitzpatrick_downloads_atlas_skips_dermaamin(
    tmp_image_dir, fitz_csv, combined_df
):
    with patch(
        "scripts.download_all_sources._download_single_image",
        side_effect=_fake_download_single,
    ):
        result = dl.download_fitzpatrick(
            combined_df, fitz_csv, tmp_image_dir, max_workers=2
        )

    assert result["downloaded"] == 2
    assert result["failed"] == 1
    assert result["skipped"] == 0
    assert os.path.exists(os.path.join(tmp_image_dir, "abc123.jpg"))
    assert os.path.exists(os.path.join(tmp_image_dir, "def456.jpg"))
    assert not os.path.exists(os.path.join(tmp_image_dir, "ghi789.jpg"))


def test_download_fitzpatrick_skips_existing(
    tmp_image_dir, fitz_csv, combined_df
):
    os.makedirs(tmp_image_dir, exist_ok=True)
    Path(os.path.join(tmp_image_dir, "abc123.jpg")).write_bytes(b"\xff\xd8" + b"\x00" * 50)

    with patch(
        "scripts.download_all_sources._download_single_image",
        side_effect=_fake_download_single,
    ):
        result = dl.download_fitzpatrick(
            combined_df, fitz_csv, tmp_image_dir, max_workers=2
        )

    assert result["skipped"] == 1
    assert result["downloaded"] == 1  # def456 only
    assert result["failed"] == 1     # dermaamin


# ---------------------------------------------------------------------------
# SCIN tests
# ---------------------------------------------------------------------------

def test_download_scin_calls_gcloud(tmp_image_dir, scin_combined_df):
    """Test that download_scin writes manifest and calls gcloud."""
    def fake_gcloud(*args, **kwargs):
        # Simulate gcloud downloading the files
        os.makedirs(tmp_image_dir, exist_ok=True)
        for img in ["111.png", "222.png", "333.png"]:
            Path(os.path.join(tmp_image_dir, img)).write_bytes(b"\x89PNG" + b"\x00" * 50)
        return MagicMock(returncode=0)

    with patch("scripts.download_all_sources.subprocess.run", side_effect=fake_gcloud) as mock_run:
        result = dl.download_scin(scin_combined_df, tmp_image_dir)

    assert mock_run.called
    # Verify gcloud was called with correct args
    call_args = mock_run.call_args[0][0]
    assert "gcloud" in call_args
    assert "storage" in call_args
    assert result["total"] == 3
    assert result["downloaded"] == 3


def test_download_scin_skips_existing(tmp_image_dir, scin_combined_df):
    os.makedirs(tmp_image_dir, exist_ok=True)
    for img in ["111.png", "222.png", "333.png"]:
        Path(os.path.join(tmp_image_dir, img)).write_bytes(b"\x89PNG" + b"\x00" * 50)

    with patch("scripts.download_all_sources.subprocess.run") as mock_run:
        result = dl.download_scin(scin_combined_df, tmp_image_dir)

    assert not mock_run.called
    assert result["skipped"] == 3
    assert result["downloaded"] == 0


# ---------------------------------------------------------------------------
# PAD-UFES tests
# ---------------------------------------------------------------------------

def test_download_pad_ufes_extracts_from_zip(tmp_path, pad_combined_df):
    """Test PAD-UFES: simulates a zip with .png files, extracts matching ones."""
    output_dir = str(tmp_path / "images")
    zip_path = tmp_path / "pad.zip"

    # Create a fake zip with the expected images
    with zf.ZipFile(str(zip_path), "w") as z:
        z.writestr("data/PAT_1_1_1.png", b"\x89PNG" + b"\x00" * 50)
        z.writestr("data/PAT_2_2_2.png", b"\x89PNG" + b"\x00" * 50)
        z.writestr("data/PAT_9_9_9.png", b"\x89PNG" + b"\x00" * 50)  # not in manifest

    with patch("scripts.download_all_sources._download_pad_zip", return_value=str(zip_path)):
        result = dl.download_pad_ufes(pad_combined_df, output_dir)

    assert result["downloaded"] == 2
    assert os.path.exists(os.path.join(output_dir, "PAT_1_1_1.png"))
    assert os.path.exists(os.path.join(output_dir, "PAT_2_2_2.png"))
    # PAT_9_9_9 should NOT be extracted (not in combined_df)
    assert not os.path.exists(os.path.join(output_dir, "PAT_9_9_9.png"))


def test_download_pad_ufes_skips_existing(tmp_path, pad_combined_df):
    output_dir = str(tmp_path / "images")
    os.makedirs(output_dir, exist_ok=True)
    for img in ["PAT_1_1_1.png", "PAT_2_2_2.png"]:
        Path(os.path.join(output_dir, img)).write_bytes(b"\x89PNG" + b"\x00" * 50)

    result = dl.download_pad_ufes(pad_combined_df, output_dir)

    assert result["skipped"] == 2
    assert result["downloaded"] == 0


# ---------------------------------------------------------------------------
# Validation tests
# ---------------------------------------------------------------------------

@pytest.fixture
def images_with_corrupt(tmp_path):
    """Dir with 2 valid images and 1 corrupt file."""
    img_dir = tmp_path / "images"
    img_dir.mkdir()

    valid = PILImage.new("RGB", (10, 10), color="red")
    valid.save(str(img_dir / "good1.jpg"))
    valid.save(str(img_dir / "good2.jpg"))

    # Corrupt file
    (img_dir / "bad.jpg").write_bytes(b"not an image")

    df = pd.DataFrame({
        "img_id": ["good1.jpg", "good2.jpg", "bad.jpg", "missing.jpg"],
        "fitzpatrick_scale": [1, 2, 3, 4],
        "source": ["fitzpatrick17k"] * 4,
    })
    return str(img_dir), df


def test_validate_downloaded_images(images_with_corrupt):
    img_dir, df = images_with_corrupt
    result = dl.validate_downloaded_images(df, img_dir)

    assert result["valid"] == 2
    assert result["corrupt"] == 1
    assert result["missing"] == 1
    assert result["total"] == 4
    assert "good1.jpg" in result["valid_ids"]
    assert "bad.jpg" in result["corrupt_ids"]
    assert "missing.jpg" in result["missing_ids"]


# ---------------------------------------------------------------------------
# Orchestrator tests
# ---------------------------------------------------------------------------

def test_download_all_returns_combined_report(tmp_path):
    """Smoke test: download_all with everything mocked returns a report."""
    output_dir = str(tmp_path / "images")
    combined_csv = tmp_path / "combined.csv"
    fitz_csv = tmp_path / "fitz.csv"

    df = pd.DataFrame({
        "img_id": ["a.jpg", "b.png"],
        "fitzpatrick_scale": [1, 2],
        "source": ["fitzpatrick17k", "scin"],
    })
    df.to_csv(str(combined_csv), index=False)
    pd.DataFrame({
        "md5hash": ["a"], "url": ["http://example.com/a.jpg"],
        "fitzpatrick_scale": [1], "fitzpatrick_centaur": [1],
        "label": ["test"], "nine_partition_label": [""],
        "three_partition_label": [""], "qc": [""],
        "url_alphanum": ["a"],
    }).to_csv(str(fitz_csv), index=False)

    mock_result = {"downloaded": 0, "failed": 0, "skipped": 0, "total": 0}
    with patch.object(dl, "download_fitzpatrick", return_value=mock_result), \
         patch.object(dl, "download_scin", return_value=mock_result), \
         patch.object(dl, "download_pad_ufes", return_value=mock_result), \
         patch.object(dl, "validate_downloaded_images", return_value={
             "valid": 0, "corrupt": 0, "missing": 2, "total": 2,
             "valid_ids": set(), "corrupt_ids": set(), "missing_ids": {"a.jpg", "b.png"},
         }):
        report = dl.download_all(
            combined_csv=str(combined_csv),
            fitz_csv_path=str(fitz_csv),
            output_dir=output_dir,
        )

    assert "fitzpatrick17k" in report
    assert "scin" in report
    assert "pad-ufes" in report
    assert "validation" in report
