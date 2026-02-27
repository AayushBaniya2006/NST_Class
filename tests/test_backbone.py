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
