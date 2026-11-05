import numpy as np
import pytest

from echospec.analysis.notch_constraints import (
    centered_notch_features,
    half_depth_residuals,
)


def test_linear_background_is_removed_without_peak_selection():
    x = np.linspace(-.5, .5, 401)
    depth = .45 * np.maximum(1 - abs(x) / .1, 0)
    p = np.stack((.7 + .1 * x - depth, .8 - .2 * x - depth * .8))
    c, d = centered_notch_features(x, p)
    np.testing.assert_allclose(c, [.45, .36], atol=1e-14)
    np.testing.assert_allclose(d, np.stack((depth, .8 * depth)), atol=1e-14)
    np.testing.assert_allclose(half_depth_residuals(x, d, c, .1), 0, atol=1e-14)
    assert np.all(half_depth_residuals(x, d, c, .12) > 0)
    assert np.all(half_depth_residuals(x, d, c, .08) < 0)


def test_single_spectrum_and_asymmetric_boundaries():
    x = np.linspace(-.5, .5, 401)
    d = .4 * np.maximum(1 - abs(x) / np.where(x < 0, .08, .12), 0)
    c, recovered = centered_notch_features(x, .7 - d)
    residuals = half_depth_residuals(x, recovered, c, .1)
    assert residuals[0] > 0 and residuals[1] < 0


def test_missing_wings_are_rejected():
    with pytest.raises(ValueError):
        centered_notch_features(np.linspace(-.1, .1, 11), np.zeros(11))


def test_one_wing_is_insufficient_even_with_many_background_points():
    x = np.linspace(-.5, .1, 121)
    with pytest.raises(ValueError, match="both spectral wings"):
        centered_notch_features(x, np.zeros_like(x))
