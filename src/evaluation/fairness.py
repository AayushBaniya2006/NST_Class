"""Fairness gap analysis across skin tone groups."""
import logging

logger = logging.getLogger(__name__)


def compute_fairness_gap(
    per_class_metrics: dict,
    metric: str = "recall",
    threshold: float = 0.15,
) -> dict:
    """Compute the fairness gap across skin tone groups.

    Gap = max(metric) - min(metric) across classes.
    Significant if gap > threshold.

    Args:
        per_class_metrics: Dict of {class_name: {metric: value}}.
        metric: Which metric to compare (default: recall).
        threshold: Gap threshold for significance.

    Returns:
        Dict with gap value, significance flag, best/worst classes.
    """
    values = {cls: m[metric] for cls, m in per_class_metrics.items()}
    max_class = max(values, key=values.get)
    min_class = min(values, key=values.get)
    gap = values[max_class] - values[min_class]

    result = {
        "metric": metric,
        "gap": round(gap, 4),
        "significant": gap > threshold,
        "threshold": threshold,
        "best_class": max_class,
        "best_value": round(values[max_class], 4),
        "worst_class": min_class,
        "worst_value": round(values[min_class], 4),
        "per_class_values": {k: round(v, 4) for k, v in values.items()},
    }

    if result["significant"]:
        logger.warning(
            f"SIGNIFICANT FAIRNESS GAP: {gap:.2%} ({metric}) — "
            f"best: {max_class} ({values[max_class]:.2%}), "
            f"worst: {min_class} ({values[min_class]:.2%})"
        )

    return result


def compare_model_fairness(
    model_per_class_metrics: dict,
    metric: str = "recall",
    threshold: float = 0.15,
) -> dict:
    """Compare fairness gaps across multiple models.

    Args:
        model_per_class_metrics: Dict of {model_name: {class: {metric: val}}}.
        metric: Metric to compare.
        threshold: Significance threshold.

    Returns:
        Dict of {model_name: fairness_gap_result}.
    """
    results = {}
    for model_name, per_class in model_per_class_metrics.items():
        results[model_name] = compute_fairness_gap(per_class, metric, threshold)

    ranked = sorted(results.items(), key=lambda x: x[1]["gap"])
    logger.info("Fairness ranking (smallest gap = most fair):")
    for i, (name, result) in enumerate(ranked):
        logger.info(f"  {i+1}. {name}: gap={result['gap']:.2%}")

    return results
