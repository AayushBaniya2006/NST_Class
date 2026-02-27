"""Pretrained backbone wrappers for feature extraction.

Supported backbones: EfficientNetV2-S, ResNet50.
Returns the backbone (with classification head removed) and its feature dimension.
"""
import torch.nn as nn
from torchvision import models


BACKBONE_REGISTRY = {
    "efficientnet_v2_s": {
        "constructor": models.efficientnet_v2_s,
        "weights": models.EfficientNet_V2_S_Weights.IMAGENET1K_V1,
    },
    "resnet50": {
        "constructor": models.resnet50,
        "weights": models.ResNet50_Weights.IMAGENET1K_V2,
    },
}


def get_backbone(name: str, pretrained: bool = True) -> tuple:
    """Load a pretrained backbone with its classification head removed.

    Args:
        name: One of 'efficientnet_v2_s' or 'resnet50'.
        pretrained: Whether to load ImageNet pretrained weights.

    Returns:
        Tuple of (backbone_model, feature_dimension).
    """
    if name not in BACKBONE_REGISTRY:
        raise ValueError(f"Unknown backbone '{name}'. Choose from: {list(BACKBONE_REGISTRY.keys())}")

    entry = BACKBONE_REGISTRY[name]
    weights = entry["weights"] if pretrained else None
    model = entry["constructor"](weights=weights)

    if name == "resnet50":
        feature_dim = model.fc.in_features
        model.fc = nn.Identity()
    elif name == "efficientnet_v2_s":
        feature_dim = model.classifier[1].in_features
        model.classifier = nn.Identity()

    return model, feature_dim


def freeze_backbone(model: nn.Module) -> None:
    """Freeze all backbone parameters."""
    for param in model.parameters():
        param.requires_grad = False


def unfreeze_last_n_blocks(model: nn.Module, name: str, n: int = 2) -> None:
    """Unfreeze the last n blocks of the backbone for fine-tuning."""
    if name == "resnet50":
        layers = [model.layer4, model.layer3][:n]
        for layer in layers:
            for param in layer.parameters():
                param.requires_grad = True
    elif name == "efficientnet_v2_s":
        blocks = list(model.features.children())
        for block in blocks[-n:]:
            for param in block.parameters():
                param.requires_grad = True
