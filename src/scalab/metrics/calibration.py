from __future__ import annotations

import numpy as np


def confidence_and_correctness(probabilities: np.ndarray, y_true: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    predictions = probabilities.argmax(axis=1)
    confidence = probabilities.max(axis=1)
    correctness = (predictions == y_true).astype(np.float64)
    return predictions, confidence, correctness


def expected_calibration_error(probabilities: np.ndarray, y_true: np.ndarray, bins: int = 15) -> float:
    _, confidence, correctness = confidence_and_correctness(probabilities, y_true)
    edges = np.linspace(0.0, 1.0, bins + 1)
    ece = 0.0
    for idx in range(bins):
        lower, upper = edges[idx], edges[idx + 1]
        mask = (confidence > lower) & (confidence <= upper) if idx > 0 else (confidence >= lower) & (confidence <= upper)
        if not np.any(mask):
            continue
        acc = float(np.mean(correctness[mask]))
        conf = float(np.mean(confidence[mask]))
        ece += (np.sum(mask) / len(confidence)) * abs(acc - conf)
    return float(ece)


def maximum_calibration_error(probabilities: np.ndarray, y_true: np.ndarray, bins: int = 15) -> float:
    _, confidence, correctness = confidence_and_correctness(probabilities, y_true)
    edges = np.linspace(0.0, 1.0, bins + 1)
    gaps = []
    for idx in range(bins):
        lower, upper = edges[idx], edges[idx + 1]
        mask = (confidence > lower) & (confidence <= upper) if idx > 0 else (confidence >= lower) & (confidence <= upper)
        if not np.any(mask):
            continue
        acc = float(np.mean(correctness[mask]))
        conf = float(np.mean(confidence[mask]))
        gaps.append(abs(acc - conf))
    return float(max(gaps)) if gaps else 0.0


def reliability_bins(probabilities: np.ndarray, y_true: np.ndarray, bins: int = 15) -> list[dict[str, float]]:
    _, confidence, correctness = confidence_and_correctness(probabilities, y_true)
    edges = np.linspace(0.0, 1.0, bins + 1)
    rows: list[dict[str, float]] = []
    for idx in range(bins):
        lower, upper = edges[idx], edges[idx + 1]
        mask = (confidence > lower) & (confidence <= upper) if idx > 0 else (confidence >= lower) & (confidence <= upper)
        if np.any(mask):
            rows.append({
                "bin_lower": float(lower),
                "bin_upper": float(upper),
                "bin_count": int(np.sum(mask)),
                "accuracy": float(np.mean(correctness[mask])),
                "confidence": float(np.mean(confidence[mask])),
            })
        else:
            rows.append({
                "bin_lower": float(lower),
                "bin_upper": float(upper),
                "bin_count": 0,
                "accuracy": 0.0,
                "confidence": 0.0,
            })
    return rows
