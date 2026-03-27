from __future__ import annotations

from scalab.config import ModelConfig
from scalab.models.base import ProbabilisticClassifier
from scalab.models.mlp import MLPDigitsClassifier


def build_model(config: ModelConfig, seed: int) -> ProbabilisticClassifier:
    if config.name == "mlp":
        return MLPDigitsClassifier(config=config, seed=seed)
    raise ValueError(f"Unsupported model type: {config.name}")
