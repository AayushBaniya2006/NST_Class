"""Training configuration as a dataclass, loadable from YAML."""
from dataclasses import dataclass, field
from typing import Optional

import yaml


@dataclass
class TrainingConfig:
    # Model
    backbone: str = "efficientnet_v2_s"
    num_classes: int = 6
    pretrained: bool = True
    dropout: float = 0.3

    # Training phases
    freeze_backbone: bool = True
    unfreeze_after_epochs: int = 5
    unfreeze_n_blocks: int = 2

    # Optimization
    epochs: int = 20
    batch_size: int = 32
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    optimizer: str = "adam"
    scheduler: str = "cosine"
    early_stopping_patience: int = 5
    use_class_weights: bool = True

    # Data
    image_size: int = 224
    num_workers: int = 4
    class_names: list = field(default_factory=lambda: ["1", "2", "3", "4", "5", "6"])

    # Logging
    wandb_project: str = "skin-tone-classifier"
    wandb_entity: Optional[str] = None
    log_every_n_steps: int = 10

    # Paths
    checkpoint_dir: str = "checkpoints"
    random_seed: int = 42

    @classmethod
    def from_yaml(cls, path: str) -> "TrainingConfig":
        """Load config from a YAML file."""
        with open(path) as f:
            raw = yaml.safe_load(f)

        # Flatten nested config
        flat = {}
        for section in raw.values():
            if isinstance(section, dict):
                flat.update(section)

        # Filter to only valid fields
        valid_fields = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in flat.items() if k in valid_fields}
        return cls(**filtered)
