"""Download images from GCS and create rotation-augmented copies."""
import argparse
import logging
import os
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from PIL import Image
from tqdm import tqdm

from src.data.prepare import _IMAGE_EXTENSIONS, _EXTENSIONS_SET

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def rotate_image(image: np.ndarray, angle: float) -> np.ndarray:
    """Rotate image by angle degrees using PIL.

    Uses affine rotation: A = [[cos θ, -sin θ], [sin θ, cos θ]]
    Per PMC5977656, θ between 10 and 175 degrees.
    Fills empty corners with black (0).

    Args:
        image: NumPy array (H, W, C) with dtype uint8.
        angle: Rotation angle in degrees.

    Returns:
        Rotated image as uint8 array, same shape as input.
    """
    pil_img = Image.fromarray(image)
    rotated = pil_img.rotate(angle, expand=False, fillcolor=(0, 0, 0))
    return np.array(rotated)


def download_from_gcs(bucket_name: str, gcs_prefix: str, local_dir: str) -> int:
    """Download images from a GCS bucket to a local directory.

    Args:
        bucket_name: GCS bucket name.
        gcs_prefix: Prefix path in the bucket (e.g. 'images/').
        local_dir: Local directory to download to.

    Returns:
        Number of files downloaded.
    """
    from google.cloud import storage

    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blobs = bucket.list_blobs(prefix=gcs_prefix)

    os.makedirs(local_dir, exist_ok=True)
    count = 0
    for blob in blobs:
        if blob.name.endswith("/"):
            continue
        filename = os.path.basename(blob.name)
        dest = os.path.join(local_dir, filename)
        if not os.path.exists(dest):
            blob.download_to_filename(dest)
            count += 1

    logger.info("Downloaded %d images from gs://%s/%s", count, bucket_name, gcs_prefix)
    return count


def process_images(
    input_dir: str,
    output_dir: str,
    angle_min: float = 10.0,
    angle_max: float = 175.0,
    copies: int = 1,
    seed: int | None = None,
) -> Tuple[int, List[Path]]:
    """Rotate all images in a directory by random angles (recursive).

    Args:
        input_dir: Directory with source images.
        output_dir: Directory to save rotated images.
        angle_min: Minimum rotation angle in degrees.
        angle_max: Maximum rotation angle in degrees.
        copies: Number of rotated copies per image.
        seed: Random seed for reproducibility.

    Returns:
        Tuple of (count of rotated images created, list of output paths).
    """
    rng = np.random.default_rng(seed)
    os.makedirs(output_dir, exist_ok=True)
    image_files = sorted(
        f for f in Path(input_dir).rglob("*")
        if f.is_file()
        and f.suffix.lower() in _EXTENSIONS_SET
        and "_rotated" not in f.stem
    )

    if not image_files:
        logger.warning("No images found in %s", input_dir)
        return 0, []

    created = 0
    output_paths: List[Path] = []
    for img_path in tqdm(image_files, desc="Rotating images"):
        try:
            img = Image.open(img_path).convert("RGB")
            arr = np.array(img)

            for c in range(copies):
                angle = rng.uniform(angle_min, angle_max)
                rotated = rotate_image(arr, angle)
                rotated_img = Image.fromarray(rotated)

                if copies == 1:
                    out_name = f"{img_path.stem}_rotated{img_path.suffix}"
                else:
                    out_name = f"{img_path.stem}_rotated_{c}{img_path.suffix}"

                out_path = Path(output_dir) / out_name
                rotated_img.save(str(out_path))
                output_paths.append(out_path)
                created += 1
        except Exception as e:
            logger.warning("Failed to process %s: %s", img_path.name, e)

    logger.info("Created %d rotated images in %s", created, output_dir)
    return created, output_paths


def upload_to_gcs(local_dir: str, bucket_name: str, gcs_prefix: str) -> int:
    """Upload files from a local directory to GCS, preserving relative paths.

    Args:
        local_dir: Local directory with files to upload.
        bucket_name: GCS bucket name.
        gcs_prefix: Destination prefix in the bucket.

    Returns:
        Number of files uploaded.
    """
    from google.cloud import storage

    client = storage.Client()
    bucket = client.bucket(bucket_name)
    local_root = Path(local_dir)
    count = 0

    for f in local_root.rglob("*"):
        if f.is_file():
            relative = f.relative_to(local_root)
            blob_name = f"{gcs_prefix}/{relative}"
            blob = bucket.blob(blob_name)
            blob.upload_from_filename(str(f))
            count += 1

    logger.info("Uploaded %d files to gs://%s/%s", count, bucket_name, gcs_prefix)
    return count


# ---------------------------------------------------------------------------
# Verification functions
# ---------------------------------------------------------------------------


def verify_download(image_dir: str, min_expected: int = 1) -> Dict[str, int]:
    """Verify downloaded images are valid.

    Args:
        image_dir: Directory containing downloaded images.
        min_expected: Minimum number of images expected.

    Returns:
        Dict with keys: total, valid, corrupt.
    """
    image_files = [
        f for f in Path(image_dir).rglob("*")
        if f.is_file() and f.suffix.lower() in _EXTENSIONS_SET
    ]

    total = len(image_files)
    valid = 0
    corrupt = 0

    for f in image_files:
        try:
            img = Image.open(f)
            img.verify()
            valid += 1
        except Exception:
            corrupt += 1
            logger.warning("Corrupt image: %s", f)

    result = {"total": total, "valid": valid, "corrupt": corrupt}
    logger.info(
        "Download verification: %d total, %d valid, %d corrupt (min expected: %d)",
        total, valid, corrupt, min_expected,
    )
    return result


def verify_rotation(
    original_dir: str,
    rotated_dir: str,
    sample_n: int = 50,
) -> Dict[str, object]:
    """Verify rotated images are valid augmentations of originals.

    Checks: every original has a _rotated counterpart, pixels differ,
    images have the same shape.

    Args:
        original_dir: Directory with original images.
        rotated_dir: Directory with rotated images.
        sample_n: Number of image pairs to sample for checks.

    Returns:
        Dict with keys: passed (bool), missing (list), checked (int).
    """
    originals = sorted(
        f for f in Path(original_dir).rglob("*")
        if f.is_file()
        and f.suffix.lower() in _EXTENSIONS_SET
        and "_rotated" not in f.stem
    )

    rotated_dir_path = Path(rotated_dir)
    rotated_stems = {f.stem for f in rotated_dir_path.rglob("*") if f.is_file()}

    missing = []
    for orig in originals:
        expected_stem = f"{orig.stem}_rotated"
        if expected_stem not in rotated_stems:
            missing.append(orig.name)

    # Sample pairs for verification
    pairs = []
    for orig in originals:
        expected_stem = f"{orig.stem}_rotated"
        candidates = list(rotated_dir_path.rglob(f"{expected_stem}.*"))
        if candidates:
            pairs.append((orig, candidates[0]))

    sample = pairs[:sample_n] if len(pairs) > sample_n else pairs

    pixels_differ = True
    shapes_match = True

    for orig_path, rotated_path in sample:
        orig_arr = np.array(Image.open(orig_path).convert("RGB"))
        rotated_arr = np.array(Image.open(rotated_path).convert("RGB"))

        if orig_arr.shape != rotated_arr.shape:
            shapes_match = False

        if np.array_equal(orig_arr, rotated_arr):
            pixels_differ = False

    passed = len(missing) == 0 and pixels_differ and shapes_match

    result = {
        "passed": passed,
        "missing": missing,
        "checked": len(sample),
    }
    logger.info("Rotation verification: passed=%s, checked=%d pairs", passed, len(sample))
    return result


def verify_upload(bucket_name: str, prefix: str, expected_count: int) -> Dict[str, object]:
    """Verify uploaded files in GCS bucket.

    Args:
        bucket_name: GCS bucket name.
        prefix: Blob prefix to check.
        expected_count: Expected number of files.

    Returns:
        Dict with keys: passed (bool), actual_count (int), expected_count (int).
    """
    from google.cloud import storage

    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blobs = list(bucket.list_blobs(prefix=prefix))
    actual = sum(1 for b in blobs if not b.name.endswith("/"))

    passed = actual >= expected_count
    result = {
        "passed": passed,
        "actual_count": actual,
        "expected_count": expected_count,
    }
    logger.info(
        "Upload verification: %d/%d files (passed=%s)",
        actual, expected_count, passed,
    )
    return result


def main():
    parser = argparse.ArgumentParser(
        description="Download images from GCS, apply rotation, optionally upload back"
    )
    # Source
    parser.add_argument("--bucket", type=str, help="GCS bucket name")
    parser.add_argument("--gcs-prefix", type=str, default="images", help="GCS prefix for source images")
    parser.add_argument("--local-input", type=str, default="data/scin/images",
                        help="Local input directory (used if --bucket not provided)")

    # Rotation parameters
    parser.add_argument("--angle-min", type=float, default=10.0, help="Min rotation angle (default: 10)")
    parser.add_argument("--angle-max", type=float, default=175.0, help="Max rotation angle (default: 175)")
    parser.add_argument("--copies", type=int, default=1, help="Rotated copies per image (default: 1)")

    # Reproducibility
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")

    # Output
    parser.add_argument("--output-dir", type=str, default="data/scin/images_rotated",
                        help="Local output directory for rotated images")
    parser.add_argument("--upload", action="store_true", help="Upload rotated images back to GCS")
    parser.add_argument("--upload-prefix", type=str, default="images_rotated",
                        help="GCS prefix for uploaded rotated images")

    # Verification
    parser.add_argument("--verify", action="store_true", help="Run verification after each step")

    args = parser.parse_args()

    # Step 1: Get source images
    if args.bucket:
        logger.info("Downloading from gs://%s/%s ...", args.bucket, args.gcs_prefix)
        download_from_gcs(args.bucket, args.gcs_prefix, args.local_input)

    if args.verify:
        dl_result = verify_download(args.local_input)
        if dl_result["valid"] == 0:
            logger.error("No valid images found after download. Aborting.")
            return

    # Step 2: Rotate images
    logger.info("Rotating images (angle=[%.1f, %.1f], copies=%d, seed=%d)",
                args.angle_min, args.angle_max, args.copies, args.seed)
    created, paths = process_images(
        args.local_input, args.output_dir, args.angle_min, args.angle_max, args.copies, args.seed
    )

    if created == 0:
        logger.error("No images processed. Check your input directory.")
        return

    if args.verify:
        rotation_result = verify_rotation(args.local_input, args.output_dir)
        if not rotation_result["passed"]:
            logger.error("Rotation verification failed: %s", rotation_result)
            return

    # Step 3: Optionally upload back to GCS
    if args.upload and args.bucket:
        logger.info("Uploading rotated images to GCS...")
        uploaded = upload_to_gcs(args.output_dir, args.bucket, args.upload_prefix)

        if args.verify:
            upload_result = verify_upload(args.bucket, args.upload_prefix, uploaded)
            if not upload_result["passed"]:
                logger.error("Upload verification failed: %s", upload_result)
                return

    logger.info("Done! %d rotated images in %s", created, args.output_dir)


if __name__ == "__main__":
    main()
