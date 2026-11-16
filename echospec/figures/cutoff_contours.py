"""Display-only smoothing of the cutoff survey's contrast boundary."""

import numpy as np
from scipy.ndimage import gaussian_filter

CONTOUR_SMOOTHING = {
    "method": "Mask-normalized Gaussian smoothing of the contour field only",
    "sigma_samples": [1.5, 3.0],
    "axis_order": ["cutoff", "amplitude"],
    "grid": "Logarithmic for the current manuscript survey",
    "edge_mode": "nearest",
    "rejected_cells": "Excluded from the average and masked again after smoothing",
    "interpretation": "Smoothed C=0.2 guides; not exact boundaries of individual cells",
    "unchanged": "Heatmaps, spectra, fits, metric arrays, threshold masks, and summaries",
}


def smooth_contrast_for_contour(contrast, accepted):
    """Suppress small contour wiggles without changing the underlying metrics.

    Arrays are ordered cutoff by amplitude. A separately filtered validity
    weight prevents rejected cells from being treated as zero contrast.
    The original mask is reapplied; no boundary is extrapolated into holes.
    """
    values = np.asarray(contrast, dtype=float)
    accepted = np.asarray(accepted, dtype=bool)
    if values.ndim != 2 or values.shape != accepted.shape:
        raise ValueError("Require matching 2D contrast and acceptance arrays")
    valid = accepted & np.isfinite(values)
    sigma = CONTOUR_SMOOTHING["sigma_samples"]
    weight = gaussian_filter(valid.astype(float), sigma, mode="nearest")
    numerator = gaussian_filter(np.where(valid, values, 0), sigma, mode="nearest")
    smoothed = np.full(values.shape, np.nan)
    np.divide(numerator, weight, out=smoothed, where=weight > 0)
    return np.ma.array(smoothed, mask=~valid)
