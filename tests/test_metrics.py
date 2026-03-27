import numpy as np

from scalab.metrics.calibration import expected_calibration_error, maximum_calibration_error


def test_perfect_calibration_metrics_zero():
    probs = np.array([
        [0.9, 0.1],
        [0.8, 0.2],
        [0.2, 0.8],
        [0.1, 0.9],
    ])
    y = np.array([0, 0, 1, 1])
    assert expected_calibration_error(probs, y, bins=5) >= 0.0
    assert maximum_calibration_error(probs, y, bins=5) >= 0.0
