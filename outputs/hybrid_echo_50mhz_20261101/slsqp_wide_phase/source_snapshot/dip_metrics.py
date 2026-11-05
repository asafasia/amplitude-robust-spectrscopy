"""Direct dip locations and finite-window half-depth widths."""

from __future__ import annotations

import numpy as np


def dip_metrics(detuning_mhz, signal):
    """Measure a central Pe dip over a sweep spanning at least ±0.5 MHz.

    Center: quadratic interpolation of the raw minimum within ±0.25 MHz.
    Width: nearest half-depth crossings of the dip after subtracting a linear
    background fitted to |detuning| >= 0.4 MHz. No Gaussian model or smoothing.
    Width depends on this explicitly chosen background and spectral window.
    """
    x, y = np.asarray(detuning_mhz), np.asarray(signal)
    if x.ndim != 1 or x.shape != y.shape or not np.all(np.diff(x) > 0):
        raise ValueError("Require matching 1D arrays with increasing detuning")
    invalid = dict(center_hz=np.nan, fwhm_hz=np.nan, contrast=np.nan, accepted=False)
    if not np.all(np.isfinite(y)) or np.ptp(y) < 1e-6:
        return invalid
    central = np.flatnonzero(abs(x) <= 0.25)
    edges = abs(x) >= 0.4
    if central.size < 3 or np.sum(edges) < 4:
        return invalid
    idx = central[np.argmin(y[central])]
    if idx in (central[0], central[-1]):
        return invalid
    a, b, _ = np.polyfit(x[idx - 1 : idx + 2] - x[idx], y[idx - 1 : idx + 2], 2)
    center = x[idx] - b / (2 * a) if a > 0 else x[idx]
    background = np.polyval(np.polyfit(x[edges], y[edges], 1), x)
    depth = background - y
    peak = central[np.argmax(depth[central])]
    contrast = depth[peak]
    if contrast < 0.001:
        return invalid
    half = contrast / 2
    left = np.flatnonzero(depth[:peak] <= half)
    right = np.flatnonzero(depth[peak + 1 :] <= half) + peak + 1
    if not left.size or not right.size:
        return invalid
    li, ri = left[-1], right[0]
    xl = x[li] + (half - depth[li]) / (depth[li + 1] - depth[li]) * (x[li + 1] - x[li])
    xr = x[ri - 1] + (half - depth[ri - 1]) / (depth[ri] - depth[ri - 1]) * (
        x[ri] - x[ri - 1]
    )
    return dict(
        center_hz=float(center * 1e6),
        fwhm_hz=float((xr - xl) * 1e6),
        contrast=float(contrast),
        accepted=True,
    )
