"""Tests for src.training.config — TrainingConfig loading."""
import warnings

import pytest
import yaml

from src.training.config import TrainingConfig


class TestTrainingConfig:
    def test_from_yaml_loads_values(self, tmp_path):
        config_data = {
            "training": {
                "backbone": "resnet50",
                "epochs": 10,
                "batch_size": 16,
                "learning_rate": 0.001,
            },
            "logging": {
                "wandb_project": "test-project",
            },
        }
        yaml_path = tmp_path / "config.yaml"
        yaml_path.write_text(yaml.dump(config_data))
        config = TrainingConfig.from_yaml(str(yaml_path))
        assert config.backbone == "resnet50"
        assert config.epochs == 10
        assert config.batch_size == 16
        assert config.learning_rate == 0.001
        assert config.wandb_project == "test-project"

    def test_from_yaml_ignores_unrecognized(self, tmp_path):
        config_data = {
            "training": {
                "backbone": "resnet50",
                "bogus_key": 42,
            },
        }
        yaml_path = tmp_path / "config.yaml"
        yaml_path.write_text(yaml.dump(config_data))
        config = TrainingConfig.from_yaml(str(yaml_path))
        assert config.backbone == "resnet50"
        assert not hasattr(config, "bogus_key")

    def test_default_values(self):
        config = TrainingConfig()
        assert config.backbone == "efficientnet_v2_s"
        assert config.num_classes == 6
        assert config.epochs == 20
        assert config.batch_size == 32
        assert config.pretrained is True
        assert config.freeze_backbone is True
        assert config.random_seed == 42
