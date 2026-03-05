"""Weights & Biases logging integration."""
import logging
from typing import Optional

logger = logging.getLogger(__name__)


def init_wandb(
    project: str,
    config: dict,
    entity: Optional[str] = None,
    run_name: Optional[str] = None,
    tags: list = None,
):
    """Initialize a W&B run.

    Args:
        project: W&B project name.
        config: Hyperparameter dict to log.
        entity: W&B team/user name.
        run_name: Optional custom run name.
        tags: Optional list of tags.

    Returns:
        W&B run object.
    """
    import wandb

    run = wandb.init(
        project=project,
        entity=entity,
        config=config,
        name=run_name,
        tags=tags or [],
    )
    logger.info(f"W&B run initialized: {run.url}")
    return run
