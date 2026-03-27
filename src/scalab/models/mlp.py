from __future__ import annotations

import numpy as np
from sklearn.neural_network import MLPClassifier

from scalab.config import ModelConfig
from scalab.models.base import PredictionBundle, ProbabilisticClassifier


class MLPDigitsClassifier(ProbabilisticClassifier):
    def __init__(self, config: ModelConfig, seed: int) -> None:
        self.model = MLPClassifier(
            hidden_layer_sizes=tuple(config.hidden_layer_sizes),
            alpha=config.alpha,
            max_iter=config.max_iter,
            early_stopping=config.early_stopping,
            random_state=seed,
        )

    def fit(self, x: np.ndarray, y: np.ndarray) -> None:
        self.model.fit(x, y)

    def predict_bundle(self, x: np.ndarray) -> PredictionBundle:
        probabilities = self.model.predict_proba(x)
        logits = np.log(np.clip(probabilities, 1e-12, 1.0))
        predictions = probabilities.argmax(axis=1)
        return PredictionBundle(logits=logits, probabilities=probabilities, predictions=predictions)
