from __future__ import annotations

from scalab.config import DatasetConfig
from scalab.datasets.base import DataSplit
from scalab.datasets.digits import load_digits_split


def load_dataset(config: DatasetConfig, seed: int) -> DataSplit:
    if config.name == "digits":
        return load_digits_split(config, seed)
    raise ValueError(f"Unsupported dataset: {config.name}")
