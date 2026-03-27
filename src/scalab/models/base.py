from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(slots=True)
class PredictionBundle:
    logits: np.ndarray
    probabilities: np.ndarray
    predictions: np.ndarray


class ProbabilisticClassifier:
    def fit(self, x: np.ndarray, y: np.ndarray) -> None:
        raise NotImplementedError

    def predict_bundle(self, x: np.ndarray) -> PredictionBundle:
        raise NotImplementedError
