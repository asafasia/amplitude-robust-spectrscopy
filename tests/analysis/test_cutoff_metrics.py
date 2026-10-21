"""Width/SNR tradeoffs reject flat and unresolved traces."""

import numpy as np
import pytest

from echospec.analysis.cutoff_metrics import cutoff_metrics


def test_gaussian_width_snr_and_unitful_ratios():
    x = np.linspace(-.5, .5, 1001)
    y = .5 - .3 * np.exp(-x*x / (2 * .04**2))
    result = cutoff_metrics(x, y)
    width = 2 * np.sqrt(2 * np.log(2)) * .04e6
    snr = .3 / np.sqrt((2*.2*.8 + 2*.5*.5) / 1000)
    assert result["accepted"]
    np.testing.assert_allclose(result["fwhm_hz"], width, rtol=1e-4)
    np.testing.assert_allclose(result["contrast_snr"], snr)
    np.testing.assert_allclose(result["snr_per_mhz"], snr / (width / 1e6), rtol=1e-4)
    np.testing.assert_allclose(result["fwhm_over_snr_hz"], width / snr, rtol=1e-4)
    np.testing.assert_allclose(result["fwhm_times_snr_hz"], width * snr, rtol=1e-4)
    more = cutoff_metrics(x, y, shots=4000)
    np.testing.assert_allclose(more["contrast_snr"], 2 * snr)
    np.testing.assert_allclose(more["shots_for_snr_10"], result["shots_for_snr_10"])


def test_flat_trace_does_not_produce_a_tradeoff():
    x = np.linspace(-.5, .5, 201)
    result = cutoff_metrics(x, np.zeros_like(x))
    assert result["status"] == "no_resolved_dip"
    assert not result["accepted"]
    assert np.isnan(result["snr_per_mhz"])


def test_undersampled_dip_keeps_raw_width_but_rejects_ranking():
    x = np.linspace(-.5, .5, 21)
    result = cutoff_metrics(x, .5 - .3 * np.exp(-x*x / (2 * .01**2)))
    assert result["status"] == "undersampled_dip"
    assert result["fwhm_hz"] > 0
    assert np.isnan(result["snr_per_mhz"])


@pytest.mark.parametrize("shots", [0, 3, 1001, 1000.5])
def test_shots_must_support_the_stated_allocation(shots):
    x = np.linspace(-.5, .5, 201)
    with pytest.raises(ValueError, match="multiple of four"):
        cutoff_metrics(x, np.full_like(x, .5), shots=shots)
