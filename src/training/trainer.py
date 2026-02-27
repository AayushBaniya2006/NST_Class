"""Training loop with class weighting, two-phase training, and early stopping."""
import logging
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score

from src.training.config import TrainingConfig

logger = logging.getLogger(__name__)


def compute_class_weights(labels: list, num_classes: int) -> list:
    """Compute inverse-frequency class weights for imbalanced data.

    Args:
        labels: List of integer labels.
        num_classes: Total number of classes.

    Returns:
        List of float weights, one per class.
    """
    counts = np.bincount(labels, minlength=num_classes).astype(float)
    counts[counts == 0] = 1  # avoid division by zero
    total = sum(counts)
    weights = [total / (num_classes * c) for c in counts]
    return weights


class EarlyStopping:
    """Stops training when validation loss doesn't improve for `patience` epochs."""

    def __init__(self, patience: int = 5):
        self.patience = patience
        self.best_loss = float("inf")
        self.counter = 0

    def step(self, val_loss: float) -> bool:
        """Returns True if training should stop."""
        if val_loss < self.best_loss:
            self.best_loss = val_loss
            self.counter = 0
            return False
        self.counter += 1
        return self.counter >= self.patience


class Trainer:
    """Two-phase trainer: frozen backbone then fine-tuned backbone.

    Args:
        model: SkinToneClassifier instance.
        config: TrainingConfig.
        train_loader: Training DataLoader.
        val_loader: Validation DataLoader.
        class_weights: Optional tensor of class weights.
        device: Torch device.
        wandb_run: Optional W&B run for logging.
    """

    def __init__(
        self,
        model: nn.Module,
        config: TrainingConfig,
        train_loader: DataLoader,
        val_loader: DataLoader,
        class_weights: Optional[torch.Tensor] = None,
        device: str = "cuda",
        wandb_run=None,
    ):
        self.model = model.to(device)
        self.config = config
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.wandb_run = wandb_run

        # Loss function
        weight = class_weights.to(device) if class_weights is not None else None
        self.criterion = nn.CrossEntropyLoss(weight=weight)

        # Optimizer — initially only head parameters if backbone is frozen
        self.optimizer = self._create_optimizer()
        phase1_t_max = config.unfreeze_after_epochs if config.freeze_backbone else config.epochs
        self.scheduler = CosineAnnealingLR(self.optimizer, T_max=phase1_t_max)
        self.early_stopping = EarlyStopping(patience=config.early_stopping_patience)

        # Checkpointing
        self.checkpoint_dir = Path(config.checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.best_val_loss = float("inf")

    def _create_optimizer(self) -> Adam:
        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        return Adam(
            trainable_params,
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )

    def _train_one_epoch(self, epoch: int) -> dict:
        self.model.train()
        running_loss = 0.0
        all_preds = []
        all_labels = []

        for batch_idx, (images, labels) in enumerate(self.train_loader):
            images = images.to(self.device)
            labels = labels.to(self.device)

            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()

            running_loss += loss.item()
            preds = outputs.argmax(dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

        avg_loss = running_loss / len(self.train_loader)
        accuracy = np.mean(np.array(all_preds) == np.array(all_labels))
        f1 = f1_score(all_labels, all_preds, average="macro")

        return {"loss": avg_loss, "accuracy": accuracy, "f1": f1}

    @torch.no_grad()
    def _validate(self) -> dict:
        self.model.eval()
        running_loss = 0.0
        all_preds = []
        all_labels = []

        for images, labels in self.val_loader:
            images = images.to(self.device)
            labels = labels.to(self.device)

            outputs = self.model(images)
            loss = self.criterion(outputs, labels)

            running_loss += loss.item()
            preds = outputs.argmax(dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

        avg_loss = running_loss / len(self.val_loader)
        accuracy = np.mean(np.array(all_preds) == np.array(all_labels))
        f1 = f1_score(all_labels, all_preds, average="macro")

        return {"loss": avg_loss, "accuracy": accuracy, "f1": f1}

    def _save_checkpoint(self, epoch: int, val_loss: float):
        path = self.checkpoint_dir / "best_model.pt"
        torch.save(self.model.state_dict(), path)
        logger.info("Saved checkpoint at epoch %d (val_loss=%.4f)", epoch, val_loss)

    def train(self) -> dict:
        """Run the full two-phase training loop.

        Phase 1: Frozen backbone (epochs 0 to unfreeze_after_epochs)
        Phase 2: Fine-tune last blocks (remaining epochs)

        Returns dict with training history.
        """
        history = {"train": [], "val": []}

        for epoch in range(self.config.epochs):
            # Phase transition: unfreeze backbone
            if epoch == self.config.unfreeze_after_epochs and self.config.freeze_backbone:
                logger.info(
                    "Epoch %d: Unfreezing last %d blocks",
                    epoch, self.config.unfreeze_n_blocks,
                )
                self.model.unfreeze_last_blocks(n=self.config.unfreeze_n_blocks)
                self.optimizer = self._create_optimizer()
                self.scheduler = CosineAnnealingLR(
                    self.optimizer,
                    T_max=self.config.epochs - epoch,
                )
                self.early_stopping = EarlyStopping(
                    patience=self.config.early_stopping_patience,
                )

            train_metrics = self._train_one_epoch(epoch)
            val_metrics = self._validate()
            self.scheduler.step()

            history["train"].append(train_metrics)
            history["val"].append(val_metrics)

            # Logging
            lr = self.optimizer.param_groups[0]["lr"]
            logger.info(
                f"Epoch {epoch+1}/{self.config.epochs} | "
                f"Train Loss: {train_metrics['loss']:.4f} Acc: {train_metrics['accuracy']:.4f} | "
                f"Val Loss: {val_metrics['loss']:.4f} Acc: {val_metrics['accuracy']:.4f} F1: {val_metrics['f1']:.4f} | "
                f"LR: {lr:.6f}"
            )

            # W&B logging
            if self.wandb_run:
                self.wandb_run.log({
                    "epoch": epoch + 1,
                    "train/loss": train_metrics["loss"],
                    "train/accuracy": train_metrics["accuracy"],
                    "train/f1": train_metrics["f1"],
                    "val/loss": val_metrics["loss"],
                    "val/accuracy": val_metrics["accuracy"],
                    "val/f1": val_metrics["f1"],
                    "learning_rate": lr,
                })

            # Checkpointing
            if val_metrics["loss"] < self.best_val_loss:
                self.best_val_loss = val_metrics["loss"]
                self._save_checkpoint(epoch, val_metrics["loss"])

            # Early stopping
            if self.early_stopping.step(val_metrics["loss"]):
                logger.info(f"Early stopping at epoch {epoch+1}")
                break

        return history
