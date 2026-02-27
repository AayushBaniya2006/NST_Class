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


def log_confusion_matrix(
    y_true: list,
    y_pred: list,
    class_names: list,
    step: Optional[int] = None,
):
    """Log a confusion matrix to W&B."""
    import wandb

    wandb.log({
        "confusion_matrix": wandb.plot.confusion_matrix(
            y_true=y_true,
            preds=y_pred,
            class_names=class_names,
        )
    }, step=step)


def log_metrics_table(metrics: dict, model_name: str):
    """Log evaluation metrics as a W&B table."""
    import wandb

    table = wandb.Table(columns=["Model", "Accuracy", "Macro F1", "ROC-AUC", "Fairness Gap"])
    table.add_data(
        model_name,
        metrics.get("accuracy"),
        metrics.get("macro_f1"),
        metrics.get("roc_auc"),
        metrics.get("fairness_gap"),
    )
    wandb.log({"evaluation_summary": table})


def log_fairness_chart(fairness_results: dict):
    """Log fairness comparison as a W&B bar chart."""
    import wandb

    data = []
    for model_name, result in fairness_results.items():
        for cls, val in result["per_class_values"].items():
            data.append([model_name, cls, val])

    table = wandb.Table(data=data, columns=["Model", "Skin Tone Group", "Recall"])
    wandb.log({
        "fairness_comparison": wandb.plot.bar(
            table, "Skin Tone Group", "Recall", title="Per-Class Recall by Model"
        )
    })
