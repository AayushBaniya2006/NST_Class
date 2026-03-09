"""Run the full augmentation ablation experiment across all buckets.

Trains the model once per augmentation bucket, saving results to separate
output directories. Each run logs to W&B with the bucket name in the run name.

Usage:
    python scripts/run_augmentation_experiment.py
    python scripts/run_augmentation_experiment.py --backbone resnet50 --no-wandb
    python scripts/run_augmentation_experiment.py --buckets flip noise combined
"""
import argparse
import logging
import subprocess
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

ALL_BUCKETS = ["control", "flip", "rotate", "crop", "noise", "combined"]


def main():
    parser = argparse.ArgumentParser(description="Run augmentation ablation experiment")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--backbone", type=str, default=None)
    parser.add_argument("--data-dir", type=str, default="data/cleaned")
    parser.add_argument("--image-dir", type=str, default="data/images")
    parser.add_argument("--output-base", type=str, default="checkpoints/aug_experiment",
                        help="Base output dir (each bucket gets a subdirectory)")
    parser.add_argument("--buckets", nargs="+", default=ALL_BUCKETS,
                        choices=ALL_BUCKETS, help="Which buckets to run (default: all)")
    parser.add_argument("--no-wandb", action="store_true")
    args = parser.parse_args()

    results = {}
    for bucket in args.buckets:
        output_dir = str(Path(args.output_base) / bucket)
        logger.info("=" * 60)
        logger.info("STARTING BUCKET: %s", bucket)
        logger.info("Output: %s", output_dir)
        logger.info("=" * 60)

        cmd = [
            sys.executable, "scripts/train.py",
            "--config", args.config,
            "--augmentation", bucket,
            "--data-dir", args.data_dir,
            "--image-dir", args.image_dir,
            "--output-dir", output_dir,
        ]
        if args.backbone:
            cmd.extend(["--backbone", args.backbone])
        if args.no_wandb:
            cmd.append("--no-wandb")

        result = subprocess.run(cmd)
        status = "SUCCESS" if result.returncode == 0 else "FAILED"
        results[bucket] = status
        logger.info("Bucket %s: %s", bucket, status)

    # Summary
    logger.info("=" * 60)
    logger.info("EXPERIMENT SUMMARY")
    logger.info("=" * 60)
    for bucket, status in results.items():
        logger.info("  %-10s %s", bucket, status)


if __name__ == "__main__":
    main()
