from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(slots=True)
class DataSplit:
    x_train: np.ndarray
    y_train: np.ndarray
    x_validation: np.ndarray
    y_validation: np.ndarray
    x_calibration: np.ndarray
    y_calibration: np.ndarray
    x_test: np.ndarray
    y_test: np.ndarray
    image_shape: tuple[int, int]
    num_classes: int
    feature_names: list[str]
