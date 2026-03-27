import numpy as np

from scalab.shifts.registry import apply_shift


def test_shift_shape_preserved():
    x = np.ones((10, 64), dtype=float)
    shifted = apply_shift("gaussian_noise", x, severity=0.2, image_shape=(8, 8), seed=7)
    assert shifted.shape == x.shape
