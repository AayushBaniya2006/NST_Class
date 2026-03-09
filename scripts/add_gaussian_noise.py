"""Download images from GCS and create Gaussian-noised augmented copies."""
import argparse
import logging
import os
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def add_gaussian_noise(image: np.ndarray, mean: float = 0.0, std: float = 25.0) -> np.ndarray:
    """Add Gaussian noise to an image array.

    Args:
        image: NumPy array (H, W, C) with dtype uint8.
        mean: Mean of the Gaussian noise.
        std: Standard deviation of the noise (higher = more noise).

    Returns:
        Noised image as uint8 array.
    """
    noise = np.random.normal(mean, std, image.shape)
    noised = image.astype(np.float64) + noise
    return np.clip(noised, 0, 255).astype(np.uint8)


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
    mean: float = 0.0,
    std: float = 25.0,
    copies: int = 1,
) -> int:
    """Add Gaussian noise to all images in a directory.

    Args:
        input_dir: Directory with source images.
        output_dir: Directory to save noised images.
        mean: Noise mean.
        std: Noise standard deviation.
        copies: Number of noised copies per image.

    Returns:
        Number of noised images created.
    """
    os.makedirs(output_dir, exist_ok=True)
    extensions = {".jpg", ".jpeg", ".png", ".bmp"}
    image_files = [
        f for f in Path(input_dir).iterdir()
        if f.suffix.lower() in extensions and f.is_file()
    ]

    if not image_files:
        logger.warning("No images found in %s", input_dir)
        return 0

    created = 0
    for img_path in tqdm(image_files, desc="Adding Gaussian noise"):
        try:
            img = Image.open(img_path).convert("RGB")
            arr = np.array(img)

            for c in range(copies):
                noised = add_gaussian_noise(arr, mean=mean, std=std)
                noised_img = Image.fromarray(noised)

                if copies == 1:
                    out_name = f"{img_path.stem}_noised{img_path.suffix}"
                else:
                    out_name = f"{img_path.stem}_noised_{c}{img_path.suffix}"

                noised_img.save(os.path.join(output_dir, out_name))
                created += 1
        except Exception as e:
            logger.warning("Failed to process %s: %s", img_path.name, e)

    logger.info("Created %d noised images in %s", created, output_dir)
    return created


def upload_to_gcs(local_dir: str, bucket_name: str, gcs_prefix: str) -> int:
    """Upload noised images back to GCS.

    Args:
        local_dir: Local directory with noised images.
        bucket_name: GCS bucket name.
        gcs_prefix: Destination prefix in the bucket.

    Returns:
        Number of files uploaded.
    """
    from google.cloud import storage

    client = storage.Client()
    bucket = client.bucket(bucket_name)
    count = 0

    for f in Path(local_dir).iterdir():
        if f.is_file():
            blob_name = f"{gcs_prefix}/{f.name}"
            blob = bucket.blob(blob_name)
            blob.upload_from_filename(str(f))
            count += 1

    logger.info("Uploaded %d files to gs://%s/%s", count, bucket_name, gcs_prefix)
    return count


def main():
    parser = argparse.ArgumentParser(
        description="Download images from GCS, add Gaussian noise, optionally upload back"
    )
    # Source
    parser.add_argument("--bucket", type=str, help="GCS bucket name")
    parser.add_argument("--gcs-prefix", type=str, default="images", help="GCS prefix for source images")
    parser.add_argument("--local-input", type=str, default="data/images",
                        help="Local input directory (used if --bucket not provided)")

    # Noise parameters
    parser.add_argument("--mean", type=float, default=0.0, help="Noise mean (default: 0.0)")
    parser.add_argument("--std", type=float, default=25.0, help="Noise std deviation (default: 25.0)")
    parser.add_argument("--copies", type=int, default=1, help="Noised copies per image (default: 1)")

    # Output
    parser.add_argument("--output-dir", type=str, default="data/images_noised",
                        help="Local output directory for noised images")
    parser.add_argument("--upload", action="store_true", help="Upload noised images back to GCS")
    parser.add_argument("--upload-prefix", type=str, default="images_noised",
                        help="GCS prefix for uploaded noised images")

    args = parser.parse_args()

    # Step 1: Get source images
    if args.bucket:
        logger.info("Downloading from gs://%s/%s ...", args.bucket, args.gcs_prefix)
        download_from_gcs(args.bucket, args.gcs_prefix, args.local_input)

    # Step 2: Add noise
    logger.info("Adding Gaussian noise (mean=%.1f, std=%.1f, copies=%d)", args.mean, args.std, args.copies)
    created = process_images(args.local_input, args.output_dir, args.mean, args.std, args.copies)

    if created == 0:
        logger.error("No images processed. Check your input directory.")
        return

    # Step 3: Optionally upload back to GCS
    if args.upload and args.bucket:
        logger.info("Uploading noised images to GCS...")
        upload_to_gcs(args.output_dir, args.bucket, args.upload_prefix)

    logger.info("Done! %d noised images in %s", created, args.output_dir)


if __name__ == "__main__":
    main()
