from __future__ import annotations

import numpy as np


def softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - np.max(logits, axis=1, keepdims=True)
    exp_vals = np.exp(shifted)
    return exp_vals / np.sum(exp_vals, axis=1, keepdims=True)


def entropy(probs: np.ndarray, axis: int = 1, eps: float = 1e-12) -> np.ndarray:
    clipped = np.clip(probs, eps, 1.0)
    return -np.sum(clipped * np.log(clipped), axis=axis)


def top2_margin(probs: np.ndarray) -> np.ndarray:
    sorted_probs = np.sort(probs, axis=1)
    return sorted_probs[:, -1] - sorted_probs[:, -2]
