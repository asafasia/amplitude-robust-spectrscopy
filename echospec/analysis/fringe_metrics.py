"""Local half-depth widths for oscillatory spectroscopy responses."""

from __future__ import annotations

import numpy as np
from scipy.interpolate import PchipInterpolator
from scipy.optimize import brentq
from scipy.signal import find_peaks


def local_fringe_metrics(detuning_mhz, population, *, center_window=(-.15, .15)):
    """Return resolved dips bracketed by maxima, with both half-depth crossings.

    The background is the mean of the two neighboring maxima. A highly
    asymmetric pair whose lower maximum never reaches half-depth is rejected.
    These are local fringe widths, not global-envelope or fitted linewidths.
    """
    x, p = np.asarray(detuning_mhz), np.asarray(population)
    if x.ndim != 1 or x.shape != p.shape or len(x) < 5:
        raise ValueError("Require matching 1D arrays with at least five points")
    if not np.all(np.isfinite(x)) or not np.all(np.isfinite(p)):
        raise ValueError("Input arrays must be finite")
    if not np.all(np.diff(x) > 0):
        raise ValueError("Detunings must strictly increase")
    minima = find_peaks(-p, prominence=1e-7)[0]
    maxima = find_peaks(p, prominence=1e-7)[0]
    interpolator = PchipInterpolator(x, p)
    result = []
    for index in minima:
        if not center_window[0] <= x[index] <= center_window[1]:
            continue
        left, right = maxima[maxima < index], maxima[maxima > index]
        if not len(left) or not len(right):
            continue
        left, right = int(left[-1]), int(right[0])
        background = float((p[left]+p[right])/2)
        depth = background-p[index]
        half = (background+p[index])/2
        if depth <= 0 or min(p[left], p[right]) <= half:
            continue
        xl = brentq(lambda z, half=half: interpolator(z)-half, x[left], x[index])
        xr = brentq(lambda z, half=half: interpolator(z)-half, x[index], x[right])
        result.append(dict(center_mhz=float(x[index]), fwhm_khz=float(1000*(xr-xl)),
                           depth=float(depth), background=background,
                           left_index=left, right_index=right))
    return result
