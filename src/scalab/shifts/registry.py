from __future__ import annotations

import numpy as np

from scalab.shifts import image


def apply_shift(name: str, x: np.ndarray, severity: float, image_shape: tuple[int, int], seed: int) -> np.ndarray:
    registry = {
        "gaussian_noise": image.gaussian_noise,
        "contrast_scale": image.contrast_scale,
        "pixel_dropout": image.pixel_dropout,
        "blur": image.blur,
        "intensity_shift": image.intensity_shift,
    }
    if name not in registry:
        raise ValueError(f"Unsupported shift family: {name}")
    shifted = registry[name](x=x, severity=severity, image_shape=image_shape, seed=seed)
    return shifted.astype(np.float64)
