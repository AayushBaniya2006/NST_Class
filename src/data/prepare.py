"""Data preparation pipeline for the Fitzpatrick17k dataset.

Handles metadata loading, label validation, grouped-label encoding,
image validation / filtering / deduplication, class-distribution
computation, stratified splitting, and image downloading.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Tuple

import imagehash
import numpy as np
import pandas as pd
import requests
from PIL import Image
from sklearn.model_selection import train_test_split
from tqdm import tqdm

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

VALID_FITZPATRICK: set[int] = {1, 2, 3, 4, 5, 6}

GROUP_MAP: dict[int, str] = {
    1: "12",
    2: "12",
    3: "34",
    4: "34",
    5: "56",
    6: "56",
}

GROUP_TO_LABEL: dict[str, int] = {
    "12": 0,
    "34": 1,
    "56": 2,
}

# Image extensions to try when locating a file on disk.
_IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _find_image_path(image_dir: str, hasher: str) -> str | None:
    """Return the first existing image path for *hasher* in *image_dir*,
    trying each extension in ``_IMAGE_EXTENSIONS``.  Returns ``None`` when
    no matching file is found.
    """
    base = Path(image_dir)
    # First check if file already has an extension
    direct = base / hasher
    if direct.is_file():
        return str(direct)
    for ext in _IMAGE_EXTENSIONS:
        candidate = base / f"{hasher}{ext}"
        if candidate.is_file():
            return str(candidate)
    return None


# ---------------------------------------------------------------------------
# 1. load_metadata
# ---------------------------------------------------------------------------

def load_metadata(csv_path: str) -> pd.DataFrame:
    """Load a Fitzpatrick17k CSV and return as a DataFrame.

    Raises ``FileNotFoundError`` if *csv_path* does not exist.
    """
    if not os.path.isfile(csv_path):
        raise FileNotFoundError(f"CSV not found: {csv_path}")
    df = pd.read_csv(csv_path)
    logger.info("Loaded %d rows, columns: %s", len(df), list(df.columns))
    return df


# ---------------------------------------------------------------------------
# 2. validate_fitzpatrick_labels
# ---------------------------------------------------------------------------

def validate_fitzpatrick_labels(df: pd.DataFrame) -> pd.DataFrame:
    """Drop rows whose ``fitzpatrick`` value is missing or not in 1-6."""
    out = df.copy()
    # Drop NaN / None
    out = out.dropna(subset=["fitzpatrick"])
    # Coerce to numeric (handles string edge cases) then filter
    out["fitzpatrick"] = pd.to_numeric(out["fitzpatrick"], errors="coerce")
    out = out.dropna(subset=["fitzpatrick"])
    out["fitzpatrick"] = out["fitzpatrick"].astype(int)
    mask = out["fitzpatrick"].isin(VALID_FITZPATRICK)
    out = out.loc[mask].reset_index(drop=True)
    logger.info("After fitzpatrick validation: %d rows", len(out))
    return out


# ---------------------------------------------------------------------------
# 3. encode_grouped_labels
# ---------------------------------------------------------------------------

def encode_grouped_labels(df: pd.DataFrame) -> pd.DataFrame:
    """Add ``skin_tone_group`` ('12'/'34'/'56') and ``skin_tone_label``
    (0/1/2) columns derived from the ``fitzpatrick`` column.
    """
    out = df.copy()
    out["skin_tone_group"] = out["fitzpatrick"].map(GROUP_MAP)
    out["skin_tone_label"] = out["skin_tone_group"].map(GROUP_TO_LABEL)
    return out


# ---------------------------------------------------------------------------
# 4. validate_images
# ---------------------------------------------------------------------------

def validate_images(
    image_dir: str,
    df: pd.DataFrame,
    hasher_col: str = "hasher",
) -> pd.DataFrame:
    """Keep only rows whose image file exists on disk and passes PIL verify."""
    keep_indices: list[int] = []
    for idx, row in df.iterrows():
        hasher = row[hasher_col]
        img_path = _find_image_path(image_dir, str(hasher))
        if img_path is None:
            continue
        try:
            with Image.open(img_path) as img:
                img.verify()
            keep_indices.append(idx)
        except Exception:
            logger.debug("Corrupted image skipped: %s", img_path)
    out = df.loc[keep_indices].reset_index(drop=True)
    logger.info("validate_images: kept %d / %d rows", len(out), len(df))
    return out


# ---------------------------------------------------------------------------
# 5. filter_human_images
# ---------------------------------------------------------------------------

def filter_human_images(
    image_dir: str,
    df: pd.DataFrame,
    hasher_col: str = "hasher",
) -> pd.DataFrame:
    """Filter out non-human images using heuristics:
    - Minimum dimensions 50x50
    - Color variance > 10 (filters diagrams / solid fills)
    - Must be RGB mode
    """
    keep_indices: list[int] = []
    for idx, row in df.iterrows():
        hasher = row[hasher_col]
        img_path = _find_image_path(image_dir, str(hasher))
        if img_path is None:
            continue
        try:
            img = Image.open(img_path)
            # Check mode
            if img.mode != "RGB":
                logger.debug("Non-RGB image skipped: %s (mode=%s)", img_path, img.mode)
                continue
            # Check dimensions
            w, h = img.size
            if w < 50 or h < 50:
                logger.debug("Too small image skipped: %s (%dx%d)", img_path, w, h)
                continue
            # Check colour variance
            arr = np.array(img, dtype=np.float64)
            variance = arr.std()
            if variance <= 10:
                logger.debug("Low variance image skipped: %s (std=%.2f)", img_path, variance)
                continue
            keep_indices.append(idx)
        except Exception:
            logger.debug("Error processing image: %s", img_path)
    out = df.loc[keep_indices].reset_index(drop=True)
    logger.info("filter_human_images: kept %d / %d rows", len(out), len(df))
    return out


# ---------------------------------------------------------------------------
# 6. deduplicate_images
# ---------------------------------------------------------------------------

def deduplicate_images(
    image_dir: str,
    df: pd.DataFrame,
    hasher_col: str = "hasher",
) -> pd.DataFrame:
    """Remove visually duplicate images using perceptual hashing (pHash)."""
    seen_hashes: dict[str, int] = {}  # phash_hex -> first index
    keep_indices: list[int] = []
    for idx, row in df.iterrows():
        hasher = row[hasher_col]
        img_path = _find_image_path(image_dir, str(hasher))
        if img_path is None:
            continue
        try:
            img = Image.open(img_path)
            phash = str(imagehash.phash(img))
            if phash not in seen_hashes:
                seen_hashes[phash] = idx
                keep_indices.append(idx)
            else:
                logger.debug(
                    "Duplicate removed: %s duplicates index %d",
                    hasher,
                    seen_hashes[phash],
                )
        except Exception:
            logger.debug("Error hashing image: %s", img_path)
    out = df.loc[keep_indices].reset_index(drop=True)
    logger.info("deduplicate_images: kept %d / %d rows", len(out), len(df))
    return out


# ---------------------------------------------------------------------------
# 7. compute_class_distribution
# ---------------------------------------------------------------------------

def compute_class_distribution(df: pd.DataFrame, column: str) -> dict:
    """Return ``{class_value: {"count": int, "percentage": float}}``."""
    if len(df) == 0:
        return {}
    counts = df[column].value_counts()
    total = len(df)
    result: dict = {}
    for cls, count in counts.items():
        result[cls] = {
            "count": int(count),
            "percentage": float(count / total * 100),
        }
    return result


# ---------------------------------------------------------------------------
# 8. generate_cleaning_report
# ---------------------------------------------------------------------------

def generate_cleaning_report(
    original_count: int,
    cleaned_df: pd.DataFrame,
    column: str,
    dropped_reasons: dict[str, int],
) -> dict:
    """Return a summary dict of the cleaning pipeline results."""
    cleaned_count = len(cleaned_df)
    return {
        "original_count": original_count,
        "cleaned_count": cleaned_count,
        "total_dropped": original_count - cleaned_count,
        "dropped_reasons": dropped_reasons,
        "class_distribution": compute_class_distribution(cleaned_df, column),
    }


# ---------------------------------------------------------------------------
# 9. stratified_split
# ---------------------------------------------------------------------------

def stratified_split(
    df: pd.DataFrame,
    label_column: str,
    ratios: Tuple[float, float, float] = (0.7, 0.15, 0.15),
    seed: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split *df* into (train, val, test) with stratification on
    *label_column*.

    ``ratios`` should sum to 1.0.  Uses ``sklearn.model_selection.train_test_split``
    twice: first to carve off the train set, then to split the remainder into
    validation and test sets.
    """
    train_ratio, val_ratio, test_ratio = ratios
    # First split: train vs (val + test)
    remaining_ratio = val_ratio + test_ratio
    train_df, temp_df = train_test_split(
        df,
        test_size=remaining_ratio,
        stratify=df[label_column],
        random_state=seed,
    )
    # Second split: val vs test (relative proportion within the remainder)
    relative_test = test_ratio / remaining_ratio
    val_df, test_df = train_test_split(
        temp_df,
        test_size=relative_test,
        stratify=temp_df[label_column],
        random_state=seed,
    )
    logger.info(
        "Split sizes — train: %d, val: %d, test: %d",
        len(train_df),
        len(val_df),
        len(test_df),
    )
    return train_df, val_df, test_df


# ---------------------------------------------------------------------------
# 10. download_images
# ---------------------------------------------------------------------------

def download_images(
    df: pd.DataFrame,
    output_dir: str,
    url_col: str = "url",
    hasher_col: str = "hasher",
) -> int:
    """Download images from URLs in *df*.  Skip files that already exist.

    Returns the number of images **newly downloaded** (not skipped).
    """
    os.makedirs(output_dir, exist_ok=True)
    downloaded = 0
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Downloading images"):
        url = row[url_col]
        hasher = str(row[hasher_col])
        # If any file for this hasher already exists, skip
        if _find_image_path(output_dir, hasher) is not None:
            continue
        # Determine extension from URL, default to .jpg
        ext = Path(url).suffix.lower()
        if ext not in _IMAGE_EXTENSIONS:
            ext = ".jpg"
        dest = os.path.join(output_dir, f"{hasher}{ext}")
        try:
            resp = requests.get(url, timeout=15)
            resp.raise_for_status()
            with open(dest, "wb") as f:
                f.write(resp.content)
            downloaded += 1
        except Exception as exc:
            logger.warning("Failed to download %s: %s", url, exc)
    logger.info("Downloaded %d new images", downloaded)
    return downloaded


# ---------------------------------------------------------------------------
# 11. run_full_pipeline
# ---------------------------------------------------------------------------

def run_full_pipeline(
    csv_path: str,
    image_dir: str,
    output_dir: str,
    seed: int = 42,
) -> dict:
    """Execute the full data-preparation pipeline and save split CSVs.

    Steps:
      1. Load metadata
      2. Validate fitzpatrick labels
      3. Encode grouped labels
      4. Validate images on disk
      5. Filter non-human images
      6. Deduplicate images
      7. Compute class distribution
      8. Generate cleaning report
      9. Stratified split  + save CSVs

    Returns a dict with keys: ``report``, ``splits`` (dict of DataFrames).
    """
    os.makedirs(output_dir, exist_ok=True)
    dropped_reasons: dict[str, int] = {}

    # 1. Load
    df = load_metadata(csv_path)
    original_count = len(df)

    # 2. Validate fitzpatrick
    df_valid = validate_fitzpatrick_labels(df)
    dropped_reasons["invalid_fitzpatrick"] = original_count - len(df_valid)

    # 3. Encode grouped labels
    df_encoded = encode_grouped_labels(df_valid)

    # 4. Validate images
    count_before = len(df_encoded)
    df_img_valid = validate_images(image_dir, df_encoded)
    dropped_reasons["missing_or_corrupt_image"] = count_before - len(df_img_valid)

    # 5. Filter non-human images
    count_before = len(df_img_valid)
    df_human = filter_human_images(image_dir, df_img_valid)
    dropped_reasons["non_human_image"] = count_before - len(df_human)

    # 6. Deduplicate
    count_before = len(df_human)
    df_dedup = deduplicate_images(image_dir, df_human)
    dropped_reasons["duplicate_image"] = count_before - len(df_dedup)

    # 7. Class distribution
    class_dist = compute_class_distribution(df_dedup, "skin_tone_label")
    logger.info("Class distribution: %s", class_dist)

    # 8. Cleaning report
    report = generate_cleaning_report(
        original_count, df_dedup, "skin_tone_label", dropped_reasons
    )

    # 9. Stratified split
    train_df, val_df, test_df = stratified_split(
        df_dedup, "skin_tone_label", seed=seed
    )
    train_df.to_csv(os.path.join(output_dir, "train.csv"), index=False)
    val_df.to_csv(os.path.join(output_dir, "val.csv"), index=False)
    test_df.to_csv(os.path.join(output_dir, "test.csv"), index=False)
    logger.info("Saved split CSVs to %s", output_dir)

    return {
        "report": report,
        "splits": {
            "train": train_df,
            "val": val_df,
            "test": test_df,
        },
    }
