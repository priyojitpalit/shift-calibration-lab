from __future__ import annotations

import numpy as np


def _reshape_images(x: np.ndarray, image_shape: tuple[int, int]) -> np.ndarray:
    return x.reshape((-1, image_shape[0], image_shape[1]))


def _flatten_images(x: np.ndarray) -> np.ndarray:
    return x.reshape((x.shape[0], -1))


def gaussian_noise(x: np.ndarray, severity: float, image_shape: tuple[int, int], seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return x + rng.normal(0.0, severity, size=x.shape)


def contrast_scale(x: np.ndarray, severity: float, image_shape: tuple[int, int], seed: int) -> np.ndarray:
    images = _reshape_images(x, image_shape)
    centered = images - images.mean(axis=(1, 2), keepdims=True)
    transformed = centered * severity + images.mean(axis=(1, 2), keepdims=True)
    return _flatten_images(transformed)


def intensity_shift(x: np.ndarray, severity: float, image_shape: tuple[int, int], seed: int) -> np.ndarray:
    return x + severity


def pixel_dropout(x: np.ndarray, severity: float, image_shape: tuple[int, int], seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    mask = rng.uniform(size=x.shape) > severity
    return x * mask


def blur(x: np.ndarray, severity: float, image_shape: tuple[int, int], seed: int) -> np.ndarray:
    images = _reshape_images(x, image_shape)
    k = int(severity)
    if k <= 0:
        return x.copy()
    kernel = np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]], dtype=np.float64)
    kernel /= kernel.sum()
    result = images.copy()
    for _ in range(k):
        padded = np.pad(result, ((0, 0), (1, 1), (1, 1)), mode="edge")
        new_imgs = np.zeros_like(result)
        for i in range(result.shape[1]):
            for j in range(result.shape[2]):
                patch = padded[:, i:i + 3, j:j + 3]
                new_imgs[:, i, j] = np.sum(patch * kernel, axis=(1, 2))
        result = new_imgs
    return _flatten_images(result)
