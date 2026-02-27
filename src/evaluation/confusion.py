"""Confusion matrix visualization."""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns


def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: list = None,
    title: str = "Confusion Matrix",
    save_path: str = None,
    normalize: bool = True,
) -> plt.Figure:
    """Plot a confusion matrix as a heatmap.

    Args:
        cm: Confusion matrix array (num_classes x num_classes).
        class_names: Labels for each class.
        title: Plot title.
        save_path: Optional path to save the figure.
        normalize: Whether to show percentages.

    Returns:
        Matplotlib Figure.
    """
    if class_names is None:
        class_names = ["Fitz I", "Fitz II", "Fitz III", "Fitz IV", "Fitz V", "Fitz VI"]

    fig, ax = plt.subplots(figsize=(8, 6))

    if normalize:
        row_sums = cm.sum(axis=1, keepdims=True).astype("float")
        row_sums[row_sums == 0] = 1  # avoid division by zero for empty classes
        cm_display = cm.astype("float") / row_sums
        fmt = ".2%"
    else:
        cm_display = cm
        fmt = "d"

    sns.heatmap(
        cm_display,
        annot=True,
        fmt=fmt,
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        ax=ax,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


def plot_fairness_comparison(
    fairness_results: dict,
    metric: str = "recall",
    save_path: str = None,
) -> plt.Figure:
    """Bar chart comparing per-class recall across models.

    Args:
        fairness_results: Output of compare_model_fairness().
        metric: Metric being compared.
        save_path: Optional path to save figure.

    Returns:
        Matplotlib Figure.
    """
    model_names = list(fairness_results.keys())
    class_names = list(fairness_results[model_names[0]]["per_class_values"].keys())

    x = np.arange(len(class_names))
    width = 0.8 / len(model_names)

    fig, ax = plt.subplots(figsize=(10, 6))

    for i, model in enumerate(model_names):
        values = [fairness_results[model]["per_class_values"][c] for c in class_names]
        gap = fairness_results[model]["gap"]
        ax.bar(x + i * width, values, width, label=f"{model} (gap={gap:.2%})")

    ax.set_xlabel("Fitzpatrick Type")
    ax.set_ylabel(metric.capitalize())
    ax.set_title(f"Per-Class {metric.capitalize()} by Model — Fairness Comparison")
    ax.set_xticks(x + width * (len(model_names) - 1) / 2)
    ax.set_xticklabels(class_names)
    ax.legend()
    ax.set_ylim(0, 1.0)
    ax.axhline(y=0.85, color="red", linestyle="--", alpha=0.3, label="Target")

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig
