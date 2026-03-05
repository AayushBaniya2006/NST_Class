import pytest
import torch
from src.models.backbone import get_backbone


class TestGetBackbone:
    def test_efficientnet_returns_model_and_feature_dim(self):
        model, feature_dim = get_backbone("efficientnet_v2_s", pretrained=False)
        assert model is not None
        assert isinstance(feature_dim, int)
        assert feature_dim > 0

    def test_resnet50_returns_model_and_feature_dim(self):
        model, feature_dim = get_backbone("resnet50", pretrained=False)
        assert model is not None
        assert isinstance(feature_dim, int)
        assert feature_dim == 2048

    def test_invalid_backbone_raises(self):
        with pytest.raises(ValueError):
            get_backbone("not_a_model", pretrained=False)

    def test_efficientnet_forward_pass(self):
        model, dim = get_backbone("efficientnet_v2_s", pretrained=False)
        x = torch.randn(2, 3, 224, 224)
        out = model(x)
        assert out.shape == (2, dim)

    def test_resnet50_forward_pass(self):
        model, dim = get_backbone("resnet50", pretrained=False)
        x = torch.randn(2, 3, 224, 224)
        out = model(x)
        assert out.shape == (2, dim)


class TestFreezeUnfreeze:
    def test_freeze_backbone_all_frozen(self):
        from src.models.backbone import freeze_backbone
        model, _ = get_backbone("resnet50", pretrained=False)
        freeze_backbone(model)
        for p in model.parameters():
            assert not p.requires_grad

    def test_unfreeze_last_n_blocks(self):
        from src.models.backbone import freeze_backbone, unfreeze_last_n_blocks
        model, _ = get_backbone("resnet50", pretrained=False)
        freeze_backbone(model)
        unfreeze_last_n_blocks(model, "resnet50", n=2)
        # layer4 and layer3 should be unfrozen
        for p in model.layer4.parameters():
            assert p.requires_grad
        for p in model.layer3.parameters():
            assert p.requires_grad
        # layer2 should remain frozen
        for p in model.layer2.parameters():
            assert not p.requires_grad
