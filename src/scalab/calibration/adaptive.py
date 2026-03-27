from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from scalab.calibration.features import build_adaptive_features
from scalab.utils.math_ops import softmax


@dataclass(slots=True)
class AdaptiveTemperatureScaler:
    learning_rate: float = 0.05
    max_iter: int = 500
    l2: float = 0.001
    include_severity: bool = False
    weights: np.ndarray | None = field(default=None, init=False)
    loss_history: list[float] = field(default_factory=list, init=False)

    def fit(self, logits: np.ndarray, probabilities: np.ndarray, x: np.ndarray, y_true: np.ndarray, severity: float = 0.0) -> "AdaptiveTemperatureScaler":
        features = build_adaptive_features(
            logits=logits,
            probabilities=probabilities,
            x=x,
            severity=severity,
            include_severity=self.include_severity,
        )
        self.weights = np.zeros(features.shape[1], dtype=np.float64)
        self.weights[0] = 0.2
        for _ in range(self.max_iter):
            loss, grad = self._loss_and_grad(features, logits, y_true)
            self.weights -= self.learning_rate * grad
            self.loss_history.append(float(loss))
        return self

    def transform(self, logits: np.ndarray, probabilities: np.ndarray, x: np.ndarray, severity: float = 0.0) -> np.ndarray:
        if self.weights is None:
            raise RuntimeError("AdaptiveTemperatureScaler must be fitted before transform().")
        features = build_adaptive_features(
            logits=logits,
            probabilities=probabilities,
            x=x,
            severity=severity,
            include_severity=self.include_severity,
        )
        temperatures = self._temperatures(features)
        scaled_logits = logits / temperatures[:, None]
        return softmax(scaled_logits)

    def temperatures(self, logits: np.ndarray, probabilities: np.ndarray, x: np.ndarray, severity: float = 0.0) -> np.ndarray:
        if self.weights is None:
            raise RuntimeError("AdaptiveTemperatureScaler must be fitted before temperatures().")
        features = build_adaptive_features(
            logits=logits,
            probabilities=probabilities,
            x=x,
            severity=severity,
            include_severity=self.include_severity,
        )
        return self._temperatures(features)

    def _temperatures(self, features: np.ndarray) -> np.ndarray:
        linear = features @ self.weights
        return 1.0 + np.log1p(np.exp(linear))

    def _loss_and_grad(self, features: np.ndarray, logits: np.ndarray, y_true: np.ndarray) -> tuple[float, np.ndarray]:
        temperatures = self._temperatures(features)
        scaled_logits = logits / temperatures[:, None]
        probs = softmax(scaled_logits)
        selected = probs[np.arange(len(y_true)), y_true]
        nll = -np.mean(np.log(np.clip(selected, 1e-12, 1.0)))
        reg = 0.5 * self.l2 * np.sum(self.weights ** 2)
        loss = nll + reg

        one_hot = np.zeros_like(probs)
        one_hot[np.arange(len(y_true)), y_true] = 1.0
        grad_scaled_logits = (probs - one_hot) / len(y_true)
        d_scaled_d_temp = -logits / (temperatures[:, None] ** 2)
        grad_temp = np.sum(grad_scaled_logits * d_scaled_d_temp, axis=1)
        sigmoid = 1.0 / (1.0 + np.exp(-(features @ self.weights)))
        grad_linear = grad_temp * sigmoid
        grad = features.T @ grad_linear + self.l2 * self.weights
        return float(loss), grad
