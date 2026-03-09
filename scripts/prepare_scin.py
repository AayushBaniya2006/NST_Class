"""Download the SCIN dataset from its public GCS bucket and prepare it
for the skin tone classification pipeline.

SCIN (Skin Condition Image Network) — 10k+ images with Fitzpatrick skin
type labels (FST1–FST6), stored at gs://dx-scin-public-data/dataset/.

Usage:
    # Download SCIN and prepare for training (local / Colab)
    python scripts/prepare_scin.py

    # Copy to your own GCS bucket too
    python scripts/prepare_scin.py --upload-bucket your-bucket-name

    # Just download CSVs to explore first
    python scripts/prepare_scin.py --csvs-only
"""
import argparse
import logging
import os
import subprocess
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Public SCIN bucket
SCIN_BUCKET = "gs://dx-scin-public-data/dataset"


def download_csvs(output_dir: str) -> dict:
    """Download SCIN CSV metadata files from the public bucket.

    Returns:
        Dict mapping csv name to local path.
    """
    os.makedirs(output_dir, exist_ok=True)
    csvs = ["scin_cases.csv", "scin_labels.csv", "scin_label_questions.csv", "scin_app_questions.csv"]
    paths = {}

    for csv in csvs:
        src = f"{SCIN_BUCKET}/{csv}"
        dest = os.path.join(output_dir, csv)
        if os.path.exists(dest):
            logger.info("Already exists: %s", dest)
        else:
            logger.info("Downloading %s ...", src)
            subprocess.run(["gsutil", "cp", src, dest], check=True)
        paths[csv] = dest

    return paths


def download_images(output_dir: str) -> int:
    """Download all SCIN images from the public bucket using gsutil -m (parallel).

    Returns:
        Number of images in the output directory after download.
    """
    os.makedirs(output_dir, exist_ok=True)
    src = f"{SCIN_BUCKET}/images/"
    logger.info("Downloading images from %s (this may take a while)...", src)
    subprocess.run(
        ["gsutil", "-m", "cp", "-r", "-n", src, output_dir],
        check=True,
    )
    # -n = no-clobber (skip existing files)
    count = sum(1 for f in Path(output_dir).rglob("*") if f.is_file())
    logger.info("Total images in %s: %d", output_dir, count)
    return count


def prepare_dataframe(cases_path: str, labels_path: str) -> pd.DataFrame:
    """Load SCIN CSVs and produce a DataFrame compatible with the
    Fitzpatrick17k pipeline (columns: hasher, fitzpatrick, skin_tone_label).

    SCIN's fitzpatrick_skin_type has values like 'FST1'...'FST6' and 'NONE_SELECTED'.
    Images are in columns image_1_path, image_2_path, image_3_path.
    """
    cases = pd.read_csv(cases_path)
    labels = pd.read_csv(labels_path)

    logger.info("SCIN cases: %d rows, labels: %d rows", len(cases), len(labels))

    # Merge cases with labels on case_id
    df = cases.merge(labels, on="case_id", how="inner")
    logger.info("After merge: %d rows", len(df))

    # Filter to rows with valid Fitzpatrick labels
    df = df[df["fitzpatrick_skin_type"].str.startswith("FST", na=False)].copy()
    logger.info("After filtering to FST1-6: %d rows", len(df))

    # Extract numeric Fitzpatrick type: 'FST3' -> 3
    df["fitzpatrick"] = df["fitzpatrick_skin_type"].str.replace("FST", "", regex=False).astype(int)

    # Encode to 0-indexed label: FST1 -> 0, FST6 -> 5
    df["skin_tone_label"] = df["fitzpatrick"] - 1

    # Explode image paths: each case can have up to 3 images
    image_cols = [c for c in df.columns if c.startswith("image_") and c.endswith("_path")]
    rows = []
    for _, row in df.iterrows():
        for col in image_cols:
            img_path = row[col]
            if pd.notna(img_path) and str(img_path).strip():
                # Use the filename (without directory) as the hasher
                hasher = Path(str(img_path)).stem
                rows.append({
                    "case_id": row["case_id"],
                    "hasher": hasher,
                    "image_filename": Path(str(img_path)).name,
                    "fitzpatrick": row["fitzpatrick"],
                    "skin_tone_label": row["skin_tone_label"],
                })

    result = pd.DataFrame(rows)
    logger.info("Final dataset: %d images from %d cases", len(result), df["case_id"].nunique())

    # Class distribution
    dist = result["skin_tone_label"].value_counts().sort_index()
    for label, count in dist.items():
        logger.info("  FST%d (label %d): %d images (%.1f%%)",
                     label + 1, label, count, 100 * count / len(result))

    return result


def split_and_save(df: pd.DataFrame, output_dir: str, seed: int = 42):
    """Stratified split into train/val/test and save CSVs."""
    from src.data.prepare import stratified_split

    os.makedirs(output_dir, exist_ok=True)

    train_df, val_df, test_df = stratified_split(
        df, label_column="skin_tone_label", seed=seed
    )

    train_df.to_csv(os.path.join(output_dir, "train.csv"), index=False)
    val_df.to_csv(os.path.join(output_dir, "val.csv"), index=False)
    test_df.to_csv(os.path.join(output_dir, "test.csv"), index=False)

    logger.info("Saved splits to %s: train=%d, val=%d, test=%d",
                 output_dir, len(train_df), len(val_df), len(test_df))


def upload_to_bucket(local_dir: str, bucket_name: str, gcs_prefix: str):
    """Upload prepared data to your own GCS bucket."""
    dest = f"gs://{bucket_name}/{gcs_prefix}"
    logger.info("Uploading %s -> %s", local_dir, dest)
    subprocess.run(
        ["gsutil", "-m", "cp", "-r", local_dir, dest],
        check=True,
    )
    logger.info("Upload complete: %s", dest)


def main():
    parser = argparse.ArgumentParser(description="Download and prepare SCIN dataset")

    parser.add_argument("--output-dir", type=str, default="data/scin",
                        help="Local directory for SCIN data (default: data/scin)")
    parser.add_argument("--cleaned-dir", type=str, default="data/scin_cleaned",
                        help="Where to save train/val/test CSVs (default: data/scin_cleaned)")
    parser.add_argument("--csvs-only", action="store_true",
                        help="Only download CSVs (skip images)")
    parser.add_argument("--skip-download", action="store_true",
                        help="Skip download, just prepare from existing files")
    parser.add_argument("--upload-bucket", type=str, default=None,
                        help="Your GCS bucket name to upload prepared data to")
    parser.add_argument("--upload-prefix", type=str, default="scin_prepared",
                        help="GCS prefix for upload (default: scin_prepared)")
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    csv_dir = args.output_dir
    image_dir = os.path.join(args.output_dir, "images")

    # Step 1: Download
    if not args.skip_download:
        csv_paths = download_csvs(csv_dir)

        if not args.csvs_only:
            download_images(image_dir)
    else:
        logger.info("Skipping download (--skip-download)")

    # Step 2: Prepare DataFrame
    cases_path = os.path.join(csv_dir, "scin_cases.csv")
    labels_path = os.path.join(csv_dir, "scin_labels.csv")

    if not os.path.exists(cases_path):
        logger.error("Missing %s — run without --skip-download first", cases_path)
        sys.exit(1)

    df = prepare_dataframe(cases_path, labels_path)

    # Step 3: Split and save
    split_and_save(df, args.cleaned_dir, seed=args.seed)

    # Step 4: Upload to your bucket (optional)
    if args.upload_bucket:
        upload_to_bucket(args.cleaned_dir, args.upload_bucket, f"{args.upload_prefix}/cleaned")
        if not args.csvs_only:
            upload_to_bucket(image_dir, args.upload_bucket, f"{args.upload_prefix}/images")

    # Print next steps
    print("\n" + "=" * 60)
    print("SCIN dataset ready!")
    print("=" * 60)
    print(f"  Images:  {image_dir}")
    print(f"  Splits:  {args.cleaned_dir}/{{train,val,test}}.csv")
    print(f"  Classes: {df['skin_tone_label'].nunique()} Fitzpatrick types")
    print(f"  Total:   {len(df)} images")
    print()
    print("To train with augmentation:")
    print(f"  python scripts/train.py --augmentation noise \\")
    print(f"    --data-dir {args.cleaned_dir} \\")
    print(f"    --image-dir {image_dir}")
    print()
    print("To run all augmentation buckets:")
    print(f"  python scripts/run_augmentation_experiment.py \\")
    print(f"    --data-dir {args.cleaned_dir} \\")
    print(f"    --image-dir {image_dir}")


if __name__ == "__main__":
    main()
