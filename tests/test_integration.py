"""Smoke test: verify the full pipeline works end-to-end on synthetic data."""
import pytest
import torch
import numpy as np
import pandas as pd
from PIL import Image
from pathlib import Path
from torch.utils.data import DataLoader

from src.data.prepare import encode_grouped_labels, stratified_split
from src.data.dataset import FitzpatrickDataset
from src.data.transforms import get_train_transforms, get_eval_transforms
from src.models.classifier import SkinToneClassifier
from src.training.config import TrainingConfig
from src.training.trainer import Trainer, compute_class_weights
from src.evaluation.metrics import compute_all_metrics
from src.evaluation.fairness import compute_fairness_gap


@pytest.fixture
def synthetic_data(tmp_path):
    """Generate a small synthetic dataset for smoke testing."""
    image_dir = tmp_path / "images"
    image_dir.mkdir()

    rows = []
    for i in range(60):
        img = Image.fromarray(np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8))
        img.save(image_dir / f"img_{i}.jpg")
        fitz = (i % 6) + 1
        rows.append({"hasher": f"img_{i}", "fitzpatrick": fitz})

    df = pd.DataFrame(rows)
    df = encode_grouped_labels(df)
    return df, str(image_dir)


def test_full_pipeline_smoke(synthetic_data):
    """Verify training -> evaluation -> fairness works end-to-end."""
    df, image_dir = synthetic_data

    # Split
    train_df, val_df, test_df = stratified_split(df, "skin_tone_label", (0.7, 0.15, 0.15))

    # Datasets
    train_ds = FitzpatrickDataset(train_df, image_dir, transform=get_train_transforms(64))
    val_ds = FitzpatrickDataset(val_df, image_dir, transform=get_eval_transforms(64))
    test_ds = FitzpatrickDataset(test_df, image_dir, transform=get_eval_transforms(64))

    train_loader = DataLoader(train_ds, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=8)
    test_loader = DataLoader(test_ds, batch_size=8)

    # Model
    config = TrainingConfig(
        backbone="resnet50",
        pretrained=False,
        image_size=64,
        epochs=2,
        batch_size=8,
        freeze_backbone=False,
        early_stopping_patience=5,
    )

    model = SkinToneClassifier("resnet50", num_classes=3, pretrained=False)

    # Class weights
    weights = compute_class_weights(train_df["skin_tone_label"].tolist(), 3)
    class_weights = torch.tensor(weights, dtype=torch.float32)

    # Train
    trainer = Trainer(model, config, train_loader, val_loader, class_weights, device="cpu")
    history = trainer.train()

    assert len(history["train"]) == 2
    assert len(history["val"]) == 2

    # Evaluate
    model.eval()
    all_preds, all_labels, all_probs = [], [], []
    with torch.no_grad():
        for images, labels in test_loader:
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)
            all_preds.extend(outputs.argmax(1).numpy())
            all_labels.extend(labels.numpy())
            all_probs.extend(probs.numpy())

    metrics = compute_all_metrics(all_labels, all_preds, np.array(all_probs))
    assert "accuracy" in metrics
    assert "per_class" in metrics

    # Fairness
    fairness = compute_fairness_gap(metrics["per_class"])
    assert "gap" in fairness
    assert "significant" in fairness
