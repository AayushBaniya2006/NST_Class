import pytest
import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader

from src.training.trainer import compute_class_weights, EarlyStopping, Trainer
from src.training.config import TrainingConfig
from src.models.classifier import SkinToneClassifier
from src.data.dataset import FitzpatrickDataset
from src.data.transforms import get_train_transforms, get_eval_transforms
from src.data.prepare import encode_labels


class TestComputeClassWeights:
    def test_balanced_classes(self):
        labels = [0] * 100 + [1] * 100 + [2] * 100 + [3] * 100 + [4] * 100 + [5] * 100
        weights = compute_class_weights(labels, num_classes=6)
        assert len(weights) == 6
        assert abs(weights[0] - weights[1]) < 0.1

    def test_imbalanced_classes(self):
        labels = [0] * 500 + [1] * 200 + [2] * 100 + [3] * 80 + [4] * 50 + [5] * 30
        weights = compute_class_weights(labels, num_classes=6)
        assert weights[5] > weights[4] > weights[3] > weights[0]


class TestEarlyStopping:
    def test_no_stop_when_improving(self):
        es = EarlyStopping(patience=3)
        assert not es.step(1.0)
        assert not es.step(0.9)
        assert not es.step(0.8)

    def test_stops_after_patience(self):
        es = EarlyStopping(patience=3)
        es.step(0.5)
        es.step(0.6)
        es.step(0.7)
        assert es.step(0.8)

    def test_resets_on_improvement(self):
        es = EarlyStopping(patience=3)
        es.step(0.5)
        es.step(0.6)
        es.step(0.4)  # improvement resets counter
        es.step(0.5)
        assert not es.step(0.6)  # only 2 non-improvements since reset


@pytest.fixture
def _synthetic_training_data(tmp_path):
    """Small synthetic dataset for trainer tests."""
    image_dir = tmp_path / "images"
    image_dir.mkdir()
    rows = []
    for i in range(60):
        img = Image.fromarray(np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8))
        img.save(image_dir / f"img_{i}.jpg")
        rows.append({"hasher": f"img_{i}", "fitzpatrick": (i % 6) + 1})
    df = encode_labels(pd.DataFrame(rows))
    train_df = df.iloc[:48]
    val_df = df.iloc[48:]
    train_ds = FitzpatrickDataset(train_df, str(image_dir), transform=get_train_transforms(64))
    val_ds = FitzpatrickDataset(val_df, str(image_dir), transform=get_eval_transforms(64))
    train_loader = DataLoader(train_ds, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=8)
    return train_df, train_loader, val_loader


class TestTrainerTraining:
    def test_trainer_loss_decreases(self, _synthetic_training_data):
        train_df, train_loader, val_loader = _synthetic_training_data
        config = TrainingConfig(
            backbone="resnet50", pretrained=False, image_size=64,
            num_classes=6, epochs=3, batch_size=8, freeze_backbone=False,
            early_stopping_patience=10,
        )
        model = SkinToneClassifier("resnet50", num_classes=6, pretrained=False)
        weights = compute_class_weights(train_df["skin_tone_label"].tolist(), 6)
        class_weights = torch.tensor(weights, dtype=torch.float32)
        trainer = Trainer(model, config, train_loader, val_loader, class_weights, device="cpu")
        history = trainer.train()
        assert history["train"][-1]["loss"] < history["train"][0]["loss"]

    def test_trainer_two_phase_training(self, _synthetic_training_data):
        train_df, train_loader, val_loader = _synthetic_training_data
        config = TrainingConfig(
            backbone="resnet50", pretrained=False, image_size=64,
            num_classes=6, epochs=3, batch_size=8, freeze_backbone=True,
            unfreeze_after_epochs=1, unfreeze_n_blocks=1,
            early_stopping_patience=10,
        )
        model = SkinToneClassifier("resnet50", num_classes=6, pretrained=False)
        model.freeze_backbone()
        # Verify backbone is frozen
        for p in model.backbone.parameters():
            assert not p.requires_grad
        trainer = Trainer(model, config, train_loader, val_loader, device="cpu")
        history = trainer.train()
        assert len(history["train"]) == 3
        # After training, backbone should be partially unfrozen
        layer4_grads = [p.requires_grad for p in model.backbone.layer4.parameters()]
        assert any(layer4_grads)
