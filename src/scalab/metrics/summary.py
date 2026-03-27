from __future__ import annotations

from scalab.metrics.calibration import expected_calibration_error, maximum_calibration_error
from scalab.metrics.classification import classification_metrics


def summarize_probabilities(probabilities, y_true, bins: int) -> dict[str, float]:
    metrics = classification_metrics(probabilities=probabilities, y_true=y_true)
    metrics["ece"] = expected_calibration_error(probabilities=probabilities, y_true=y_true, bins=bins)
    metrics["mce"] = maximum_calibration_error(probabilities=probabilities, y_true=y_true, bins=bins)
    metrics["mean_confidence"] = float(probabilities.max(axis=1).mean())
    return metrics
