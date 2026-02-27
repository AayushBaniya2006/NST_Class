import pytest
import torch
from src.models.classifier import SkinToneClassifier


class TestSkinToneClassifier:
    def test_forward_shape(self):
        model = SkinToneClassifier("resnet50", num_classes=6, pretrained=False)
        x = torch.randn(4, 3, 224, 224)
        out = model(x)
        assert out.shape == (4, 6)

    def test_efficientnet_forward_shape(self):
        model = SkinToneClassifier("efficientnet_v2_s", num_classes=6, pretrained=False)
        x = torch.randn(4, 3, 224, 224)
        out = model(x)
        assert out.shape == (4, 6)

    def test_freeze_backbone(self):
        model = SkinToneClassifier("resnet50", num_classes=6, pretrained=False)
        model.freeze_backbone()
        for param in model.backbone.parameters():
            assert not param.requires_grad
        # Classifier head should still be trainable
        for param in model.head.parameters():
            assert param.requires_grad

    def test_unfreeze_backbone(self):
        model = SkinToneClassifier("resnet50", num_classes=6, pretrained=False)
        model.freeze_backbone()
        model.unfreeze_last_blocks(n=2)
        trainable = sum(1 for p in model.backbone.parameters() if p.requires_grad)
        assert trainable > 0
