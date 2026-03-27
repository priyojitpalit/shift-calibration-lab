from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from scalab.utils.math_ops import softmax


@dataclass(slots=True)
class ScalarTemperatureScaler:
    temperature: float = 1.0

    def fit(self, logits: np.ndarray, y_true: np.ndarray, search_bounds: tuple[float, float], grid_points: int = 400) -> "ScalarTemperatureScaler":
        lower, upper = search_bounds
        temps = np.linspace(lower, upper, grid_points)
        losses = [self._nll(logits / temp, y_true) for temp in temps]
        self.temperature = float(temps[int(np.argmin(losses))])
        return self

    def transform(self, logits: np.ndarray) -> np.ndarray:
        return softmax(logits / self.temperature)

    @staticmethod
    def _nll(logits: np.ndarray, y_true: np.ndarray) -> float:
        probs = softmax(logits)
        selected = probs[np.arange(len(y_true)), y_true]
        return float(-np.mean(np.log(np.clip(selected, 1e-12, 1.0))))
