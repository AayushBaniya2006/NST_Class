"""Tests for src/utils/logging.py."""
import sys
from unittest.mock import MagicMock

import pytest


@pytest.fixture(autouse=True)
def mock_wandb():
    """Mock wandb module for all tests in this file."""
    mock = MagicMock()
    mock.init.return_value = MagicMock(url="https://wandb.ai/test")
    sys.modules["wandb"] = mock
    yield mock
    sys.modules.pop("wandb", None)


class TestInitWandb:
    def test_calls_wandb_init(self, mock_wandb):
        import importlib
        import src.utils.logging
        importlib.reload(src.utils.logging)

        src.utils.logging.init_wandb(project="test-project", config={"lr": 0.001})

        mock_wandb.init.assert_called_once_with(
            project="test-project",
            entity=None,
            config={"lr": 0.001},
            name=None,
            tags=[],
        )

    def test_passes_entity(self, mock_wandb):
        import importlib
        import src.utils.logging
        importlib.reload(src.utils.logging)

        src.utils.logging.init_wandb(project="p", config={}, entity="my-team")

        _, kwargs = mock_wandb.init.call_args
        assert kwargs["entity"] == "my-team"

    def test_default_tags_empty_list(self, mock_wandb):
        import importlib
        import src.utils.logging
        importlib.reload(src.utils.logging)

        src.utils.logging.init_wandb(project="p", config={})

        _, kwargs = mock_wandb.init.call_args
        assert kwargs["tags"] == []
