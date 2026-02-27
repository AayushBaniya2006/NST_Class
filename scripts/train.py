"""CLI entrypoint for training — usable locally or as Vertex AI custom job."""
import argparse
import logging
import sys
from pathlib import Path

import torch
import pandas as pd
from torch.utils.data import DataLoader

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.dataset import FitzpatrickDataset
from src.data.transforms import get_train_transforms, get_eval_transforms
from src.models.classifier import SkinToneClassifier
from src.training.config import TrainingConfig
from src.training.trainer import Trainer, compute_class_weights

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Train skin tone classifier")
    parser.add_argument("--config", type=str, default="configs/default.yaml", help="Path to config YAML")
    parser.add_argument("--backbone", type=str, default=None, help="Override backbone name")
    parser.add_argument("--epochs", type=int, default=None, help="Override epochs")
    parser.add_argument("--batch-size", type=int, default=None, help="Override batch size")
    parser.add_argument("--lr", type=float, default=None, help="Override learning rate")
    parser.add_argument("--data-dir", type=str, default="data/cleaned", help="Directory with train/val CSVs")
    parser.add_argument("--image-dir", type=str, default="data/images", help="Directory with images")
    parser.add_argument("--output-dir", type=str, default="checkpoints", help="Where to save model")
    parser.add_argument("--no-wandb", action="store_true", help="Disable W&B logging")
    args = parser.parse_args()

    # Load config
    config = TrainingConfig.from_yaml(args.config)

    # Apply overrides
    if args.backbone:
        config.backbone = args.backbone
    if args.epochs:
        config.epochs = args.epochs
    if args.batch_size:
        config.batch_size = args.batch_size
    if args.lr:
        config.learning_rate = args.lr

    config.checkpoint_dir = args.output_dir

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Device: {device}")
    logger.info(f"Config: {config}")

    # Load data
    train_df = pd.read_csv(f"{args.data_dir}/train.csv")
    val_df = pd.read_csv(f"{args.data_dir}/val.csv")

    train_dataset = FitzpatrickDataset(train_df, args.image_dir, transform=get_train_transforms(config.image_size))
    val_dataset = FitzpatrickDataset(val_df, args.image_dir, transform=get_eval_transforms(config.image_size))

    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=config.num_workers, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False, num_workers=config.num_workers, pin_memory=True)

    # Class weights
    class_weights = None
    if config.use_class_weights:
        weights = compute_class_weights(train_df["skin_tone_label"].tolist(), num_classes=config.num_classes)
        class_weights = torch.tensor(weights, dtype=torch.float32)
        logger.info(f"Class weights: {weights}")

    # Model
    model = SkinToneClassifier(config.backbone, config.num_classes, pretrained=config.pretrained)
    if config.freeze_backbone:
        model.freeze_backbone()

    # W&B
    wandb_run = None
    if not args.no_wandb:
        from src.utils.logging import init_wandb
        wandb_run = init_wandb(config.wandb_project, vars(config), run_name=f"{config.backbone}_vertex")

    # Train
    trainer = Trainer(model, config, train_loader, val_loader, class_weights, device, wandb_run)
    history = trainer.train()

    # Save final model
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    final_path = f"{args.output_dir}/{config.backbone}_final.pt"
    torch.save(model.state_dict(), final_path)
    logger.info(f"Final model saved to {final_path}")

    if wandb_run:
        import wandb
        wandb.finish()


if __name__ == "__main__":
    main()
