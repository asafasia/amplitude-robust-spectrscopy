"""Local dip fitting must recover widths without half-depth branch switches."""

import numpy as np
import pytest

from echospec.analysis.dip_metrics import dip_metrics
from echospec.analysis.robust_dip_fit import evaluate_dip_fit, robust_dip_fit


@pytest.mark.parametrize("center,width", [(0.0, .07), (.025, .18), (-.02, .4)])
def test_recovers_gaussian_on_sloping_curved_background(center, width):
    x = np.linspace(-.5, .5, 201)
    y = .5 + .07 * x + .2 * x*x - .25 * np.exp(-4*np.log(2)*((x-center)/width)**2)
    fit = robust_dip_fit(x, y)
    assert fit["accepted"], fit
    np.testing.assert_allclose(fit["fwhm_hz"], width * 1e6, rtol=1e-6)
    np.testing.assert_allclose(fit["contrast"], .25, rtol=1e-6)
    np.testing.assert_allclose(fit["center_hz"], center * 1e6, atol=.01)
    np.testing.assert_allclose(evaluate_dip_fit(x, fit), y, atol=1e-7)


def test_small_side_structure_does_not_change_central_width_abruptly():
    x = np.linspace(-.5, .5, 1001)
    widths, crossing_widths = [], []
    for curvature in np.linspace(.15, .17, 9):
        y = (.393 + curvature*x*x - .03*np.exp(-4*np.log(2)*(x/.065)**2)
             - .002*np.exp(-.5*((abs(x)-.13)/.022)**2))
        fit = robust_dip_fit(x, y)
        assert fit["accepted"], fit
        widths.append(fit["fwhm_hz"])
        crossing_widths.append(dip_metrics(x, y)["fwhm_hz"])
    assert np.max(abs(np.diff(crossing_widths))) > 90000
    assert np.max(abs(np.diff(widths))) / np.mean(widths) < .01
    np.testing.assert_allclose(widths, 65000, rtol=.06)


def test_flat_and_unresolved_spectra_are_rejected():
    x = np.linspace(-.5, .5, 201)
    assert not robust_dip_fit(x, np.full_like(x, .5))["accepted"]
    narrow = .5 - .2*np.exp(-4*np.log(2)*(x/.01)**2)
    assert not robust_dip_fit(x, narrow)["accepted"]


def test_invalid_coordinate_arrays_are_rejected():
    with pytest.raises(ValueError, match="increasing"):
        robust_dip_fit([0, 1, 0], [0, .1, .2])
