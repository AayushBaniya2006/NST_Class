"""Generate augmented copies of minority-class images using Albumentations.

Allowed transforms (per professor guidance):
- HorizontalFlip
- Rotate (±30 degrees)
- RandomResizedCrop (scale 0.8-1.0)
- GaussNoise

FORBIDDEN: brightness, contrast, color jitter, blurring.
"""
import argparse
import logging
import os
import random
from pathlib import Path
from typing import Dict, List, Tuple

import albumentations as A
import numpy as np
from PIL import Image
from tqdm import tqdm

from src.data.prepare import _IMAGE_EXTENSIONS

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

_EXTENSIONS_SET = {ext.lower() for ext in _IMAGE_EXTENSIONS}


def get_augmentation_pipeline(image_size: int = 224) -> A.Compose:
    """Return the approved Albumentations augmentation pipeline.

    Only includes transforms approved by professor:
    flips, rotations, random crop/zoom, Gaussian noise.
    NO brightness, contrast, color jitter, or blurring.
    """
    return A.Compose([
        A.HorizontalFlip(p=0.5),
        A.Rotate(limit=30, border_mode=0, value=0, p=0.8),
        A.RandomResizedCrop(
            size=(image_size, image_size),
            scale=(0.8, 1.0),
            ratio=(0.9, 1.1),
            p=0.7,
        ),
        A.GaussNoise(std_range=(0.02, 0.06), p=0.5),
    ])


def augment_images(
    input_dir: str,
    output_dir: str,
    img_ids: List[str],
    target_count: int,
    seed: int = 42,
    image_size: int = 224,
) -> Tuple[int, List[Path]]:
    """Generate augmented copies of specified images.

    Cycles through img_ids applying random augmentations until
    target_count new images have been created.

    Args:
        input_dir: Directory containing source images.
        output_dir: Directory to save augmented images.
        img_ids: List of image filenames to augment (e.g. ["abc.jpg", "def.png"]).
        target_count: Number of augmented images to generate.
        seed: Random seed for reproducibility.
        image_size: Output image size (square).

    Returns:
        Tuple of (count created, list of output paths).
    """
    rng = np.random.default_rng(seed)
    os.makedirs(output_dir, exist_ok=True)
    pipeline = get_augmentation_pipeline(image_size)

    # Resolve actual paths for the img_ids
    input_path = Path(input_dir)
    resolved = []
    for img_id in img_ids:
        candidate = input_path / img_id
        if candidate.is_file():
            resolved.append(candidate)
        else:
            stem = Path(img_id).stem
            for ext in _EXTENSIONS_SET:
                candidate = input_path / f"{stem}{ext}"
                if candidate.is_file():
                    resolved.append(candidate)
                    break

    if not resolved:
        logger.warning("No source images found in %s for %d img_ids", input_dir, len(img_ids))
        return 0, []

    logger.info("Found %d/%d source images to augment", len(resolved), len(img_ids))

    created = 0
    output_paths: List[Path] = []
    cycle_idx = 0

    with tqdm(total=target_count, desc="Augmenting images") as pbar:
        while created < target_count:
            img_path = resolved[cycle_idx % len(resolved)]
            cycle_idx += 1

            try:
                with Image.open(img_path) as img:
                    image = img.convert("RGB")
                arr = np.array(image)

                # Seed albumentations per-image for reproducibility
                aug_seed = int(rng.integers(0, 2**31))
                random.seed(aug_seed)
                np.random.seed(aug_seed % (2**31))
                augmented = pipeline(image=arr)["image"]

                out_name = f"{img_path.stem}_aug_{created}{img_path.suffix}"
                out_path = Path(output_dir) / out_name
                Image.fromarray(augmented).save(str(out_path))
                output_paths.append(out_path)
                created += 1
                pbar.update(1)
            except Exception as e:
                logger.warning("Failed to augment %s: %s", img_path.name, e)

    logger.info("Created %d augmented images in %s", created, output_dir)
    return created, output_paths


def verify_augmentation(
    original_dir: str,
    augmented_dir: str,
    sample_n: int = 50,
) -> Dict[str, object]:
    """Verify augmented images are valid.

    Checks: images are openable, not corrupt, and pixels differ
    from the source image (confirming augmentation was applied).

    Args:
        original_dir: Directory with original images.
        augmented_dir: Directory with augmented images.
        sample_n: Number of augmented images to sample for checks.

    Returns:
        Dict with keys: passed (bool), total (int), checked (int),
        valid (int), corrupt (int).
    """
    aug_files = sorted(
        f for f in Path(augmented_dir).rglob("*")
        if f.is_file() and f.suffix.lower() in _EXTENSIONS_SET
    )

    total = len(aug_files)
    sample = aug_files[:sample_n] if len(aug_files) > sample_n else aug_files

    valid = 0
    corrupt = 0
    pixels_differ = True

    for f in sample:
        try:
            img = Image.open(f)
            img.verify()
            valid += 1

            # Check pixels differ from source (strip _aug_N suffix)
            stem = f.stem
            orig_stem = stem.rsplit("_aug_", 1)[0] if "_aug_" in stem else stem
            orig_candidates = list(Path(original_dir).rglob(f"{orig_stem}.*"))
            if orig_candidates:
                orig_arr = np.array(Image.open(orig_candidates[0]).convert("RGB"))
                aug_arr = np.array(Image.open(f).convert("RGB"))
                if orig_arr.shape == aug_arr.shape and np.array_equal(orig_arr, aug_arr):
                    pixels_differ = False
        except Exception:
            corrupt += 1
            logger.warning("Corrupt augmented image: %s", f)

    passed = corrupt == 0 and total > 0 and pixels_differ
    result = {
        "passed": passed,
        "total": total,
        "checked": len(sample),
        "valid": valid,
        "corrupt": corrupt,
    }
    logger.info(
        "Augmentation verification: %d total, %d/%d valid (passed=%s)",
        total, valid, len(sample), passed,
    )
    return result


def main():
    parser = argparse.ArgumentParser(
        description="Augment minority-class images using Albumentations"
    )
    parser.add_argument("--input-dir", type=str, required=True,
                        help="Directory with source images")
    parser.add_argument("--output-dir", type=str, required=True,
                        help="Directory to save augmented images")
    parser.add_argument("--csv", type=str, required=True,
                        help="Path to combined_dataset.csv")
    parser.add_argument("--target-class", type=int, required=True,
                        help="Fitzpatrick scale class to augment (1-6)")
    parser.add_argument("--target-count", type=int, required=True,
                        help="Number of augmented images to generate")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed (default: 42)")
    parser.add_argument("--image-size", type=int, default=224,
                        help="Output image size (default: 224)")
    parser.add_argument("--verify", action="store_true",
                        help="Run verification after augmentation")

    args = parser.parse_args()

    import pandas as pd
    df = pd.read_csv(args.csv)
    class_imgs = df[df["fitzpatrick_scale"] == args.target_class]["img_id"].tolist()

    logger.info("Class %d: %d source images, generating %d augmented",
                args.target_class, len(class_imgs), args.target_count)

    created, paths = augment_images(
        args.input_dir, args.output_dir, class_imgs,
        args.target_count, args.seed, args.image_size,
    )

    if args.verify and created > 0:
        result = verify_augmentation(args.input_dir, args.output_dir)
        if not result["passed"]:
            logger.error("Verification failed: %s", result)
            return

    logger.info("Done! %d augmented images for class %d", created, args.target_class)


if __name__ == "__main__":
    main()
