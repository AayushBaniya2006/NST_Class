"""Full classifier: pretrained backbone + classification head."""
import torch.nn as nn

from src.models.backbone import get_backbone, freeze_backbone, unfreeze_last_n_blocks


class SkinToneClassifier(nn.Module):
    """Skin tone classifier with pretrained backbone and FC head.

    Args:
        backbone_name: Name of the backbone ('resnet50' or 'efficientnet_v2_s').
        num_classes: Number of output classes (default 3 for grouped Fitzpatrick).
        pretrained: Whether to load ImageNet pretrained weights.
        dropout: Dropout probability before the final FC layer.
    """

    def __init__(
        self,
        backbone_name: str,
        num_classes: int = 3,
        pretrained: bool = True,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.backbone_name = backbone_name
        self.backbone, feature_dim = get_backbone(backbone_name, pretrained=pretrained)
        self.head = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(feature_dim, num_classes),
        )

    def forward(self, x):
        features = self.backbone(x)
        return self.head(features)

    def freeze_backbone(self):
        """Freeze all backbone parameters."""
        freeze_backbone(self.backbone)

    def unfreeze_last_blocks(self, n: int = 2):
        """Unfreeze the last n blocks for fine-tuning."""
        unfreeze_last_n_blocks(self.backbone, self.backbone_name, n=n)
