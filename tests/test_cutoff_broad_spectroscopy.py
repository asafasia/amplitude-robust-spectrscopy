import numpy as np

from scripts.make_cutoff_broad_spectroscopy import peak_centers


def test_smoothed_peak_preserves_symmetric_line_center_and_rejects_zero_drive():
    frequency = np.linspace(-5, 5, 2001)
    expected_center = 0.233
    gaussian = np.exp(-0.5 * ((frequency - expected_center) / 0.7) ** 2)
    spectra = np.stack((np.zeros_like(frequency), gaussian))
    for sigma in (0.25, 0.5, 1.0):
        result = peak_centers(frequency, spectra, sigma)
        assert np.isnan(result[0])
        np.testing.assert_allclose(result[1], expected_center, atol=1e-5)
