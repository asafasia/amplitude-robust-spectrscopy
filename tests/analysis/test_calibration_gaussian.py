"""Synthetic checks of the campaign-compatible linewidth estimator."""

import numpy as np
import pytest

from echospec.analysis.calibration_gaussian import (
    FWHM_FACTOR,
    fit_calibration_dip,
    negative_gaussian,
)


def test_recovers_gaussian_width_with_documented_smoothing():
    x = np.linspace(-200000, 200000, 200)
    sigma = 12000.0
    y = negative_gaussian(x, 0.55, 0.10, 20000, sigma)
    result = fit_calibration_dip(x, y)
    expected = FWHM_FACTOR * np.sqrt(sigma**2 + (x[1] - x[0]) ** 2)
    assert result["accepted"]
    assert result["center_hz"] == pytest.approx(20000, abs=1)
    assert result["fwhm_hz"] == pytest.approx(expected, rel=0.001)


def test_frequency_alignment_does_not_change_linewidth():
    x = np.linspace(-200000, 200000, 200)
    y = negative_gaussian(x, 0.55, 0.10, 20000, 18000)
    original = fit_calibration_dip(x, y)
    centered = fit_calibration_dip(x - 20000, y)
    assert centered["fwhm_hz"] == pytest.approx(original["fwhm_hz"], rel=1e-5)
    assert centered["center_hz"] == pytest.approx(0, abs=1)


def test_flat_trace_has_no_reported_width():
    result = fit_calibration_dip(np.linspace(-200000, 200000, 200), np.ones(200))
    assert not result["accepted"]
    assert np.isnan(result["fwhm_hz"])
