from __future__ import annotations

import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from scalab.config import DatasetConfig
from scalab.datasets.base import DataSplit


def load_digits_split(config: DatasetConfig, seed: int) -> DataSplit:
    dataset = load_digits()
    x = dataset.data.astype(np.float64)
    y = dataset.target.astype(np.int64)

    x_train, x_temp, y_train, y_temp = train_test_split(
        x,
        y,
        test_size=1.0 - config.split.train,
        random_state=seed,
        stratify=y,
    )

    remaining = config.split.validation + config.split.calibration + config.split.test
    val_ratio = config.split.validation / remaining
    cal_ratio = config.split.calibration / (config.split.calibration + config.split.test)

    x_validation, x_rest, y_validation, y_rest = train_test_split(
        x_temp,
        y_temp,
        test_size=1.0 - val_ratio,
        random_state=seed + 1,
        stratify=y_temp,
    )

    x_calibration, x_test, y_calibration, y_test = train_test_split(
        x_rest,
        y_rest,
        test_size=1.0 - cal_ratio,
        random_state=seed + 2,
        stratify=y_rest,
    )

    if config.normalize:
        scaler = StandardScaler()
        x_train = scaler.fit_transform(x_train)
        x_validation = scaler.transform(x_validation)
        x_calibration = scaler.transform(x_calibration)
        x_test = scaler.transform(x_test)

    return DataSplit(
        x_train=x_train,
        y_train=y_train,
        x_validation=x_validation,
        y_validation=y_validation,
        x_calibration=x_calibration,
        y_calibration=y_calibration,
        x_test=x_test,
        y_test=y_test,
        image_shape=(8, 8),
        num_classes=len(np.unique(y)),
        feature_names=[f"pixel_{i}" for i in range(x.shape[1])],
    )
