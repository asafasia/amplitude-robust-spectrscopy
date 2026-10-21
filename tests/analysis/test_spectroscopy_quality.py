"""Shot allocation and low-contrast behavior of visibility metrics."""

import numpy as np

from echospec.analysis.spectroscopy_quality import spectroscopy_quality


def test_projection_noise_and_shot_scaling():
    x = np.linspace(-.5, .5, 1001)
    y = .5 - .3 * np.exp(-x*x / (2 * .04**2))
    result = spectroscopy_quality(x, y)
    expected = .3 / np.sqrt((2*.2*.8 + 2*.5*.5) / 1000)
    np.testing.assert_allclose(result["contrast_snr"], expected, rtol=1e-10)
    np.testing.assert_allclose(result["fwhm_hz"],
                               2*np.sqrt(2*np.log(2))*.04e6, rtol=1e-4)
    more = spectroscopy_quality(x, y, shots=4000)
    np.testing.assert_allclose(more["contrast_snr"], 2*result["contrast_snr"])


def test_flat_signal_is_not_a_resolved_dip():
    x = np.linspace(-.5, .5, 201)
    result = spectroscopy_quality(x, np.full_like(x, .2))
    assert not result["accepted"]
    assert np.isnan(result["contrast_snr"])
