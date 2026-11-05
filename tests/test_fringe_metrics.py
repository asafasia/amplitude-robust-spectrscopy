import numpy as np
import pytest

from echospec.analysis.fringe_metrics import local_fringe_metrics


def test_local_fringe_width_and_contrast_match_shifted_cosine():
    x = np.linspace(-.5, .5, 8001)
    period, amplitude = .0786, .061
    p = .47-amplitude*np.cos(2*np.pi*(x-.0113)/period)
    result = local_fringe_metrics(x, p)
    assert len(result) >= 3
    np.testing.assert_allclose([r["fwhm_khz"] for r in result], period*500, atol=.001)
    np.testing.assert_allclose([r["depth"] for r in result], 2*amplitude, atol=2e-6)


def test_incomplete_asymmetric_fringe_is_rejected_without_crossing_error():
    x = np.linspace(-.1, .1, 9)
    p = np.array([.8, 1., 0., .1, .05, .8, .9, .85, .8])
    assert local_fringe_metrics(x, p) == []


def test_flat_monotonic_and_invalid_frequency_axes():
    x = np.linspace(-.2, .2, 101)
    assert local_fringe_metrics(x, np.full_like(x, .5)) == []
    assert local_fringe_metrics(x, .5+x) == []
    with pytest.raises(ValueError, match="increase"):
        local_fringe_metrics(x[::-1], np.full_like(x, .5))
