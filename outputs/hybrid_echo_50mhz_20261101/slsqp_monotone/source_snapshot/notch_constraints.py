"""Continuous spectral quantities for constrained echo optimization.

These fixed-center quantities avoid selecting extrema and half-height roots
inside a gradient calculation. Exact extracted FWHM and dip contrast must still
be checked separately, especially for asymmetric or nonmonotone spectra.
"""

import numpy as np


def centered_notch_features(detuning_mhz, probability):
    """Return fixed-zero depth and detrended depletion for one or more spectra."""
    x = np.asarray(detuning_mhz, float)
    p = np.asarray(probability, float)
    if x.ndim != 1 or len(x) < 5 or np.any(np.diff(x) <= 0):
        raise ValueError("Detuning must be a strictly increasing 1D grid")
    if p.shape[-1] != len(x) or not np.isfinite(p).all():
        raise ValueError("Finite spectra must match the detuning grid")
    edges = abs(x) >= .4 - 1e-12
    if edges.sum() < 4 or not x[0] < 0 < x[-1]:
        raise ValueError("Need both spectral wings and zero inside the grid")
    matrix = np.stack((x, np.ones_like(x)), axis=1)
    coefficients = np.linalg.lstsq(matrix[edges], p.reshape(-1, len(x))[:, edges].T,
                                   rcond=None)[0]
    background = (matrix @ coefficients).T.reshape(p.shape)
    depletion = background - p
    contrast = np.array([np.interp(0, x, row) for row in depletion.reshape(-1, len(x))])
    return contrast.reshape(p.shape[:-1]), depletion


def half_depth_residuals(detuning_mhz, depletion, contrast, widths_mhz):
    """Positive values place both trial boundaries outside half depth.

    This implies a width bound for a connected central dip with its maximum
    at zero. It is a smooth optimization surrogate, not the final FWHM test.
    """
    x, d = np.asarray(detuning_mhz), np.asarray(depletion)
    c = np.asarray(contrast)
    widths = np.broadcast_to(widths_mhz, c.shape)
    if d.shape[:-1] != c.shape or np.any(widths <= 0):
        raise ValueError("Contrasts and positive widths must match spectra")
    left = np.array([np.interp(-w / 2, x, row) for row, w in
                     zip(d.reshape(-1, len(x)), widths.ravel(), strict=True)])
    right = np.array([np.interp(w / 2, x, row) for row, w in
                      zip(d.reshape(-1, len(x)), widths.ravel(), strict=True)])
    return np.stack((c.ravel() / 2 - left, c.ravel() / 2 - right), axis=-1).reshape(c.shape + (2,))
