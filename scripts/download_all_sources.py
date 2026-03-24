"""Download images from all 3 dataset sources into a single local directory.

Sources:
  1. Fitzpatrick17k — GCS bucket gs://fitzpatrick-dataset/all_images/ (primary),
     with HTTP URL fallback for any missing images
  2. SCIN — public GCS bucket gs://dx-scin-public-data/dataset/images/
  3. PAD-UFES — Mendeley Data zip download

Usage (CLI):
    python scripts/download_all_sources.py --output-dir data/combined_images

Usage (from notebook):
    from scripts.download_all_sources import download_all
    report = download_all("data/combined_images")
"""
import argparse
import logging
import os
import subprocess
import sys
import tempfile
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, Optional

import pandas as pd
import requests
from PIL import Image
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Fitzpatrick17k
# ---------------------------------------------------------------------------

FITZ_BUCKET = "gs://fitzpatrick-dataset/all_images"


def _download_single_image(args: tuple) -> bool:
    """Download a single image from URL. Returns True on success."""
    url, dest = args
    try:
        resp = requests.get(url, timeout=15, allow_redirects=True)
        resp.raise_for_status()
        content_type = resp.headers.get("content-type", "")
        if "image" not in content_type and len(resp.content) < 1000:
            return False
        with open(dest, "wb") as f:
            f.write(resp.content)
        return True
    except Exception:
        return False


def _download_fitz_from_gcs(needed: set, output_dir: str) -> int:
    """Download Fitzpatrick17k images from GCS bucket. Returns count downloaded."""
    manifest_path = os.path.join(tempfile.gettempdir(), "fitz_manifest.txt")
    with open(manifest_path, "w") as f:
        for img_id in sorted(needed):
            f.write(f"{FITZ_BUCKET}/{img_id}\n")

    env = os.environ.copy()
    env["CLOUDSDK_STORAGE_THREAD_COUNT"] = "16"
    env["CLOUDSDK_STORAGE_PROCESS_COUNT"] = "4"

    try:
        with open(manifest_path, "r") as manifest_f:
            subprocess.run(
                ["gcloud", "storage", "cp", "-I", f"{output_dir}/"],
                stdin=manifest_f,
                check=True,
                env=env,
            )
    except subprocess.CalledProcessError as e:
        logger.warning("Fitzpatrick17k GCS download had errors: %s", e)
    except FileNotFoundError:
        logger.warning("Fitzpatrick17k: 'gcloud' CLI not found, skipping GCS.")
        return 0

    # Count how many we got
    new_existing = {
        f.name for f in Path(output_dir).iterdir() if f.is_file()
    }
    downloaded = len(new_existing & needed)
    return downloaded


def download_fitzpatrick(
    combined_df: pd.DataFrame,
    fitz_csv_path: str,
    output_dir: str,
    max_workers: int = 50,
    batch_size: int = 500,
) -> Dict[str, int]:
    """Download Fitzpatrick17k images from GCS bucket, with URL fallback.

    Strategy:
      1. Try GCS bucket gs://fitzpatrick-dataset/all_images/ (fast, bulk)
      2. Fall back to original HTTP URLs for any remaining images

    Args:
        combined_df: DataFrame with img_id and source columns.
        fitz_csv_path: Path to fitzpatrick17k.csv (has md5hash -> url mapping).
        output_dir: Directory to save images.
        max_workers: Concurrent download threads (URL fallback).
        batch_size: Images per progress batch (URL fallback).

    Returns:
        Dict with keys: downloaded, failed, skipped, total.
    """
    os.makedirs(output_dir, exist_ok=True)

    fitz_ids = set(
        combined_df[combined_df["source"] == "fitzpatrick17k"]["img_id"]
    )

    # Check what we already have
    existing = {
        f.name for f in Path(output_dir).iterdir() if f.is_file()
    } if os.path.isdir(output_dir) else set()
    needed = fitz_ids - existing
    skipped = len(fitz_ids) - len(needed)

    if not needed:
        logger.info("Fitzpatrick17k: all %d images already present", len(fitz_ids))
        return {"downloaded": 0, "failed": 0, "skipped": skipped, "total": len(fitz_ids)}

    # --- Phase 1: GCS bucket (fast, bulk) ---
    logger.info(
        "Fitzpatrick17k: downloading %d images from GCS bucket...", len(needed),
    )
    gcs_downloaded = _download_fitz_from_gcs(needed, output_dir)
    logger.info("Fitzpatrick17k GCS: %d downloaded", gcs_downloaded)

    # Recheck what's still needed after GCS
    existing = {
        f.name for f in Path(output_dir).iterdir() if f.is_file()
    }
    still_needed = fitz_ids - existing

    if not still_needed:
        total_dl = gcs_downloaded
        logger.info("Fitzpatrick17k done: %d from GCS, %d skipped", total_dl, skipped)
        return {"downloaded": total_dl, "failed": 0, "skipped": skipped, "total": len(fitz_ids)}

    # --- Phase 2: URL fallback for remaining images ---
    logger.info(
        "Fitzpatrick17k: %d still missing, trying original URLs...", len(still_needed),
    )
    fitz_csv = pd.read_csv(fitz_csv_path)
    fitz_csv["img_id"] = fitz_csv["md5hash"] + ".jpg"
    url_map = dict(zip(fitz_csv["img_id"], fitz_csv["url"]))

    tasks = []
    for img_id in still_needed:
        url = url_map.get(img_id)
        if url and pd.notna(url):
            dest = os.path.join(output_dir, img_id)
            tasks.append((str(url), dest))

    url_downloaded = 0
    url_failed = 0

    if tasks:
        for batch_start in range(0, len(tasks), batch_size):
            batch = tasks[batch_start : batch_start + batch_size]
            batch_num = batch_start // batch_size + 1
            total_batches = (len(tasks) + batch_size - 1) // batch_size

            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = {executor.submit(_download_single_image, t): t for t in batch}
                for future in tqdm(
                    as_completed(futures),
                    total=len(batch),
                    desc=f"Fitz URL batch {batch_num}/{total_batches}",
                ):
                    if future.result():
                        url_downloaded += 1
                    else:
                        url_failed += 1

    total_dl = gcs_downloaded + url_downloaded
    total_failed = url_failed + (len(still_needed) - len(tasks))  # no-URL entries
    logger.info(
        "Fitzpatrick17k done: %d from GCS, %d from URLs, %d failed, %d skipped",
        gcs_downloaded, url_downloaded, total_failed, skipped,
    )
    return {"downloaded": total_dl, "failed": total_failed, "skipped": skipped, "total": len(fitz_ids)}


# ---------------------------------------------------------------------------
# SCIN
# ---------------------------------------------------------------------------

SCIN_BUCKET = "gs://dx-scin-public-data/dataset/images"


def download_scin(
    combined_df: pd.DataFrame,
    output_dir: str,
) -> Dict[str, int]:
    """Download SCIN images from the public GCS bucket.

    Uses `gcloud storage cp` with a manifest file for targeted downloads.
    Requires `gcloud` CLI to be authenticated.

    Args:
        combined_df: DataFrame with img_id and source columns.
        output_dir: Directory to save images.

    Returns:
        Dict with keys: downloaded, failed, skipped, total.
    """
    os.makedirs(output_dir, exist_ok=True)

    scin_ids = set(
        combined_df[combined_df["source"] == "scin"]["img_id"]
    )

    existing = {
        f.name for f in Path(output_dir).iterdir() if f.is_file()
    } if os.path.isdir(output_dir) else set()
    needed = scin_ids - existing
    skipped = len(scin_ids) - len(needed)

    if not needed:
        logger.info("SCIN: all %d images already present", len(scin_ids))
        return {"downloaded": 0, "failed": 0, "skipped": skipped, "total": len(scin_ids)}

    logger.info("SCIN: downloading %d images (%d skipped)...", len(needed), skipped)

    # Write manifest of GCS paths for targeted download
    manifest_path = os.path.join(tempfile.gettempdir(), "scin_manifest.txt")
    with open(manifest_path, "w") as f:
        for img_id in sorted(needed):
            f.write(f"{SCIN_BUCKET}/{img_id}\n")

    # Use gcloud storage cp with manifest piped via stdin (no shell=True)
    env = os.environ.copy()
    env["CLOUDSDK_STORAGE_THREAD_COUNT"] = "16"
    env["CLOUDSDK_STORAGE_PROCESS_COUNT"] = "4"

    try:
        with open(manifest_path, "r") as manifest_f:
            subprocess.run(
                ["gcloud", "storage", "cp", "-I", f"{output_dir}/"],
                stdin=manifest_f,
                check=True,
                env=env,
            )
    except subprocess.CalledProcessError as e:
        logger.error("SCIN gcloud download failed: %s", e)
    except FileNotFoundError:
        logger.error("SCIN: 'gcloud' CLI not found. Install Google Cloud SDK.")

    # Count results
    new_existing = {
        f.name for f in Path(output_dir).iterdir() if f.is_file()
    }
    downloaded = len((new_existing & scin_ids) - existing)
    failed = len(needed) - downloaded

    logger.info("SCIN done: %d downloaded, %d failed, %d skipped", downloaded, failed, skipped)
    return {"downloaded": downloaded, "failed": failed, "skipped": skipped, "total": len(scin_ids)}


# ---------------------------------------------------------------------------
# PAD-UFES
# ---------------------------------------------------------------------------

PAD_UFES_URLS = [
    "https://data.mendeley.com/public-files/datasets/zr7vgbcyr2/files/"
    "757370e4-3a5a-4219-ae1c-cf1acfe4e22d/file_downloaded",
    "https://prod-dcd-datasets-cache-zipfiles.s3.eu-west-1.amazonaws.com/zr7vgbcyr2-1.zip",
]


def _download_pad_zip(tmp_dir: str) -> Optional[str]:
    """Download the PAD-UFES zip to tmp_dir. Returns path or None."""
    zip_path = os.path.join(tmp_dir, "pad_ufes.zip")
    if os.path.exists(zip_path) and os.path.getsize(zip_path) > 1_000_000:
        logger.info("PAD-UFES zip already downloaded: %s", zip_path)
        return zip_path

    for url in PAD_UFES_URLS:
        logger.info("PAD-UFES: trying %s ...", url[:80])
        try:
            resp = requests.get(url, timeout=120, stream=True)
            resp.raise_for_status()
            with open(zip_path, "wb") as f:
                for chunk in resp.iter_content(chunk_size=8192):
                    f.write(chunk)
            if os.path.getsize(zip_path) > 1_000_000:
                logger.info("PAD-UFES zip downloaded: %.1f MB", os.path.getsize(zip_path) / 1e6)
                return zip_path
        except Exception as e:
            logger.warning("PAD-UFES download failed from %s: %s", url[:60], e)

    logger.error(
        "PAD-UFES: could not download from any URL. "
        "Manual download: https://data.mendeley.com/datasets/zr7vgbcyr2/1"
    )
    return None


def download_pad_ufes(
    combined_df: pd.DataFrame,
    output_dir: str,
) -> Dict[str, int]:
    """Download PAD-UFES images from Mendeley Data zip.

    Downloads the zip archive, extracts only the images listed in
    combined_df, and copies them to output_dir.

    Args:
        combined_df: DataFrame with img_id and source columns.
        output_dir: Directory to save images.

    Returns:
        Dict with keys: downloaded, failed, skipped, total.
    """
    os.makedirs(output_dir, exist_ok=True)

    pad_ids = set(
        combined_df[combined_df["source"] == "pad-ufes"]["img_id"]
    )

    existing = {
        f.name for f in Path(output_dir).iterdir() if f.is_file()
    } if os.path.isdir(output_dir) else set()
    needed = pad_ids - existing
    skipped = len(pad_ids) - len(needed)

    if not needed:
        logger.info("PAD-UFES: all %d images already present", len(pad_ids))
        return {"downloaded": 0, "failed": 0, "skipped": skipped, "total": len(pad_ids)}

    logger.info("PAD-UFES: need %d images (%d skipped)...", len(needed), skipped)

    # Download zip
    tmp_dir = os.path.join(tempfile.gettempdir(), "pad_ufes")
    os.makedirs(tmp_dir, exist_ok=True)
    zip_path = _download_pad_zip(tmp_dir)

    if zip_path is None:
        return {"downloaded": 0, "failed": len(needed), "skipped": skipped, "total": len(pad_ids)}

    # Extract only needed images
    downloaded = 0
    with zipfile.ZipFile(zip_path, "r") as zf:
        # Build a map of filename -> zip path for all .png files
        png_entries = {}
        for entry in zf.namelist():
            name = Path(entry).name
            if name.lower().endswith(".png") and name in needed:
                png_entries[name] = entry

        logger.info("PAD-UFES: found %d/%d matching images in zip", len(png_entries), len(needed))

        for name, entry in tqdm(png_entries.items(), desc="Extracting PAD-UFES"):
            data = zf.read(entry)
            dest = os.path.join(output_dir, name)
            with open(dest, "wb") as f:
                f.write(data)
            downloaded += 1

    failed = len(needed) - downloaded
    logger.info("PAD-UFES done: %d extracted, %d missing from zip, %d skipped", downloaded, failed, skipped)
    return {"downloaded": downloaded, "failed": failed, "skipped": skipped, "total": len(pad_ids)}


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def validate_downloaded_images(
    combined_df: pd.DataFrame,
    image_dir: str,
    sample_n: int = 0,
) -> Dict[str, object]:
    """Validate that downloaded images exist and are openable.

    Args:
        combined_df: DataFrame with img_id column.
        image_dir: Directory containing images.
        sample_n: If > 0, only validate a random sample. 0 = validate all.

    Returns:
        Dict with valid, corrupt, missing counts and ID sets.
    """
    all_ids = list(combined_df["img_id"])
    if sample_n > 0 and sample_n < len(all_ids):
        import random
        all_ids = random.sample(all_ids, sample_n)

    existing_files = {
        f.name for f in Path(image_dir).iterdir() if f.is_file()
    } if os.path.isdir(image_dir) else set()

    valid_ids = set()
    corrupt_ids = set()
    missing_ids = set()

    for img_id in tqdm(all_ids, desc="Validating images"):
        if img_id not in existing_files:
            missing_ids.add(img_id)
            continue
        try:
            img_path = os.path.join(image_dir, img_id)
            with Image.open(img_path) as img:
                img.verify()
            valid_ids.add(img_id)
        except Exception:
            corrupt_ids.add(img_id)

    result = {
        "valid": len(valid_ids),
        "corrupt": len(corrupt_ids),
        "missing": len(missing_ids),
        "total": len(all_ids),
        "valid_ids": valid_ids,
        "corrupt_ids": corrupt_ids,
        "missing_ids": missing_ids,
    }
    logger.info(
        "Validation: %d valid, %d corrupt, %d missing (of %d)",
        result["valid"], result["corrupt"], result["missing"], result["total"],
    )
    return result


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def download_all(
    combined_csv: str = "combined_dataset.csv",
    fitz_csv_path: str = "data/fitzpatrick17k.csv",
    output_dir: str = "data/combined_images",
    max_workers: int = 50,
    upload_bucket: Optional[str] = None,
    upload_prefix: str = "combined_images",
) -> Dict[str, dict]:
    """Download images from all 3 sources and validate.

    Args:
        combined_csv: Path to combined_dataset.csv.
        fitz_csv_path: Path to fitzpatrick17k.csv.
        output_dir: Local directory for all images.
        max_workers: Concurrent download threads (Fitzpatrick17k).
        upload_bucket: If set, upload images to this GCS bucket after download.
        upload_prefix: GCS prefix for upload.

    Returns:
        Dict with per-source results and validation summary.
    """
    df = pd.read_csv(combined_csv)
    logger.info("Combined dataset: %d images from %d sources", len(df), df["source"].nunique())
    logger.info("By source:\n%s", df["source"].value_counts().to_string())

    report = {}

    # 1. Fitzpatrick17k
    logger.info("\n" + "=" * 60)
    logger.info("FITZPATRICK17K")
    logger.info("=" * 60)
    report["fitzpatrick17k"] = download_fitzpatrick(
        df, fitz_csv_path, output_dir, max_workers=max_workers,
    )

    # 2. SCIN
    logger.info("\n" + "=" * 60)
    logger.info("SCIN")
    logger.info("=" * 60)
    report["scin"] = download_scin(df, output_dir)

    # 3. PAD-UFES
    logger.info("\n" + "=" * 60)
    logger.info("PAD-UFES")
    logger.info("=" * 60)
    report["pad-ufes"] = download_pad_ufes(df, output_dir)

    # 4. Validate
    logger.info("\n" + "=" * 60)
    logger.info("VALIDATION")
    logger.info("=" * 60)
    report["validation"] = validate_downloaded_images(df, output_dir)

    # 5. Optional GCS upload
    if upload_bucket:
        logger.info("\n" + "=" * 60)
        logger.info("UPLOADING TO GCS")
        logger.info("=" * 60)
        gcs_dest = f"gs://{upload_bucket}/{upload_prefix}"
        env = os.environ.copy()
        env["CLOUDSDK_STORAGE_THREAD_COUNT"] = "16"
        env["CLOUDSDK_STORAGE_PROCESS_COUNT"] = "4"
        subprocess.run(
            ["gcloud", "storage", "cp", "-n", f"{output_dir}/*", f"{gcs_dest}/"],
            env=env,
            check=True,
        )
        logger.info("Uploaded to %s", gcs_dest)

    # 6. Print summary
    _print_summary(df, report)

    return report


def _print_summary(df: pd.DataFrame, report: Dict[str, dict]):
    """Print a human-readable summary table."""
    print("\n" + "=" * 60)
    print("DOWNLOAD SUMMARY")
    print("=" * 60)

    for source in ["fitzpatrick17k", "scin", "pad-ufes"]:
        r = report[source]
        total_in_csv = len(df[df["source"] == source])
        print(f"\n  {source}:")
        print(f"    In CSV:     {total_in_csv}")
        print(f"    Downloaded: {r['downloaded']}")
        print(f"    Failed:     {r['failed']}")
        print(f"    Skipped:    {r['skipped']}")

    v = report["validation"]
    print(f"\n  VALIDATION:")
    print(f"    Valid:   {v['valid']}")
    print(f"    Corrupt: {v['corrupt']}")
    print(f"    Missing: {v['missing']}")
    print(f"    Total:   {v['total']}")

    print(f"\n  USABLE IMAGES: {v['valid']}/{v['total']}")

    # Per-class breakdown of valid images
    valid_df = df[df["img_id"].isin(v["valid_ids"])]
    if not valid_df.empty:
        print(f"\n  PER-CLASS (valid only):")
        for cls in sorted(valid_df["fitzpatrick_scale"].unique()):
            count = len(valid_df[valid_df["fitzpatrick_scale"] == cls])
            print(f"    Type {cls}: {count}")

    print("=" * 60)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Download images from all 3 dataset sources"
    )
    parser.add_argument(
        "--combined-csv", type=str, default="combined_dataset.csv",
        help="Path to combined_dataset.csv (default: combined_dataset.csv)",
    )
    parser.add_argument(
        "--fitz-csv", type=str, default="data/fitzpatrick17k.csv",
        help="Path to fitzpatrick17k.csv (default: data/fitzpatrick17k.csv)",
    )
    parser.add_argument(
        "--output-dir", type=str, default="data/combined_images",
        help="Local directory for images (default: data/combined_images)",
    )
    parser.add_argument(
        "--max-workers", type=int, default=50,
        help="Concurrent download threads for HTTP (default: 50)",
    )
    parser.add_argument(
        "--upload-bucket", type=str, default=None,
        help="GCS bucket to upload images to after download",
    )
    parser.add_argument(
        "--upload-prefix", type=str, default="combined_images",
        help="GCS prefix for upload (default: combined_images)",
    )
    parser.add_argument(
        "--validate-only", action="store_true",
        help="Skip downloads, just validate existing images",
    )

    args = parser.parse_args()

    if args.validate_only:
        df = pd.read_csv(args.combined_csv)
        result = validate_downloaded_images(df, args.output_dir)
        _print_summary(df, {
            "fitzpatrick17k": {"downloaded": 0, "failed": 0, "skipped": 0, "total": 0},
            "scin": {"downloaded": 0, "failed": 0, "skipped": 0, "total": 0},
            "pad-ufes": {"downloaded": 0, "failed": 0, "skipped": 0, "total": 0},
            "validation": result,
        })
    else:
        download_all(
            combined_csv=args.combined_csv,
            fitz_csv_path=args.fitz_csv,
            output_dir=args.output_dir,
            max_workers=args.max_workers,
            upload_bucket=args.upload_bucket,
            upload_prefix=args.upload_prefix,
        )


if __name__ == "__main__":
    main()
