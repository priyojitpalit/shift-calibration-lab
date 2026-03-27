import numpy as np

from scalab.calibration.temperature import ScalarTemperatureScaler


def test_scalar_temperature_bounds():
    logits = np.array([[3.0, 1.0], [2.5, 0.1], [0.1, 2.9], [0.2, 2.5]])
    y = np.array([0, 0, 1, 1])
    scaler = ScalarTemperatureScaler().fit(logits, y, search_bounds=(0.5, 5.0), grid_points=50)
    assert 0.5 <= scaler.temperature <= 5.0
