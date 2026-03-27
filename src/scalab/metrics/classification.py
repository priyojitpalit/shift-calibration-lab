from __future__ import annotations

import numpy as np
from sklearn.metrics import accuracy_score, log_loss


def multiclass_brier_score(probabilities: np.ndarray, y_true: np.ndarray) -> float:
    one_hot = np.zeros_like(probabilities)
    one_hot[np.arange(len(y_true)), y_true] = 1.0
    return float(np.mean(np.sum((probabilities - one_hot) ** 2, axis=1)))


def classification_metrics(probabilities: np.ndarray, y_true: np.ndarray) -> dict[str, float]:
    predictions = probabilities.argmax(axis=1)
    return {
        "accuracy": float(accuracy_score(y_true, predictions)),
        "nll": float(log_loss(y_true, probabilities, labels=np.arange(probabilities.shape[1]))),
        "brier": multiclass_brier_score(probabilities, y_true),
    }
