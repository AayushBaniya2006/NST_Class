"""Google Cloud Storage upload/download helpers."""
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def upload_directory_to_gcs(local_dir: str, bucket_name: str, gcs_prefix: str) -> int:
    """Upload a local directory to GCS.

    Args:
        local_dir: Local directory path.
        bucket_name: GCS bucket name.
        gcs_prefix: Prefix path in the bucket.

    Returns:
        Number of files uploaded.
    """
    from google.cloud import storage

    client = storage.Client()
    bucket = client.bucket(bucket_name)
    local_path = Path(local_dir)
    count = 0

    for file_path in local_path.rglob("*"):
        if file_path.is_file():
            blob_name = f"{gcs_prefix}/{file_path.relative_to(local_path)}"
            blob = bucket.blob(blob_name)
            blob.upload_from_filename(str(file_path))
            count += 1

    logger.info(f"Uploaded {count} files to gs://{bucket_name}/{gcs_prefix}/")
    return count


def upload_file_to_gcs(local_path: str, bucket_name: str, blob_name: str) -> str:
    """Upload a single file to GCS.

    Returns:
        GCS URI of the uploaded file.
    """
    from google.cloud import storage

    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(blob_name)
    blob.upload_from_filename(local_path)
    uri = f"gs://{bucket_name}/{blob_name}"
    logger.info(f"Uploaded {local_path} to {uri}")
    return uri


def download_file_from_gcs(bucket_name: str, blob_name: str, local_path: str) -> str:
    """Download a single file from GCS."""
    from google.cloud import storage

    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(blob_name)
    Path(local_path).parent.mkdir(parents=True, exist_ok=True)
    blob.download_to_filename(local_path)
    logger.info(f"Downloaded gs://{bucket_name}/{blob_name} to {local_path}")
    return local_path


def generate_automl_csv(
    df,
    image_gcs_prefix: str,
    split_column: str = "split",
    label_column: str = "skin_tone_group",
    hasher_column: str = "hasher",
    output_path: str = "automl_manifest.csv",
) -> str:
    """Generate a CSV manifest for Vertex AI AutoML Image Classification.

    Format: ML_USE,GCS_FILE_PATH,LABEL

    Args:
        df: DataFrame with split, label, and hasher columns.
        image_gcs_prefix: GCS prefix where images are stored.
        split_column: Column indicating train/val/test split.
        label_column: Column with the class label.
        hasher_column: Column with the image filename (without extension).
        output_path: Where to save the manifest CSV.

    Returns:
        Path to the saved manifest.
    """
    split_map = {"train": "TRAINING", "val": "VALIDATION", "test": "TEST"}

    rows = []
    for _, row in df.iterrows():
        ml_use = split_map.get(row[split_column], "TRAINING")
        gcs_path = f"{image_gcs_prefix}/{row[hasher_column]}.jpg"
        label = str(row[label_column])
        rows.append(f"{ml_use},{gcs_path},{label}")

    with open(output_path, "w") as f:
        f.write("\n".join(rows))

    logger.info(f"Generated AutoML manifest with {len(rows)} rows at {output_path}")
    return output_path
