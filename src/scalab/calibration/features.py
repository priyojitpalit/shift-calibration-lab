from __future__ import annotations

import numpy as np

from scalab.utils.math_ops import entropy, top2_margin


def build_adaptive_features(
    logits: np.ndarray,
    probabilities: np.ndarray,
    x: np.ndarray,
    severity: float = 0.0,
    include_severity: bool = False,
) -> np.ndarray:
    top_conf = probabilities.max(axis=1)
    margin = top2_margin(probabilities)
    ent = entropy(probabilities) / np.log(probabilities.shape[1])
    mean_abs = np.mean(np.abs(x), axis=1)
    variance = np.var(x, axis=1)
    l2_norm = np.linalg.norm(x, axis=1) / np.sqrt(x.shape[1])
    max_logit = np.max(logits, axis=1)

    features = np.column_stack([
        np.ones(len(x)),
        top_conf,
        margin,
        ent,
        mean_abs,
        variance,
        l2_norm,
        max_logit,
    ])
    if include_severity:
        severity_column = np.full((len(x), 1), severity, dtype=np.float64)
        features = np.hstack([features, severity_column])
    return features.astype(np.float64)
