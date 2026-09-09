"""Bounded Gaussian dip estimator matching the OPX beta-calibration analysis."""

from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.optimize import curve_fit

FWHM_FACTOR = 2 * np.sqrt(2 * np.log(2))


def negative_gaussian(x, offset, depth, center, sigma):
    return offset - depth * np.exp(-0.5 * ((x - center) / sigma) ** 2)


def fit_calibration_dip(detuning_hz: np.ndarray, signal: np.ndarray) -> dict:
    """Fit a trace with the source campaign's preprocessing and acceptance.

    Trim up to 40 samples at either edge, smooth with a one-sample Gaussian,
    then fit offset-depth*exp(-(f-center)^2/(2*sigma^2)). Covariance errors
    describe this smoothed fit, not an independent measurement uncertainty.
    """
    finite = np.isfinite(detuning_hz) & np.isfinite(signal)
    x = np.asarray(detuning_hz, dtype=float)[finite]
    y = np.asarray(signal, dtype=float)[finite]
    invalid = {
        key: np.nan
        for key in (
            "center_hz",
            "fwhm_hz",
            "center_error_hz",
            "fwhm_error_hz",
            "r_squared",
            "contrast",
        )
    }
    invalid["accepted"] = False
    if x.size < 5 or np.ptp(y) <= 0:
        return invalid
    if not np.all(np.diff(x) > 0):
        raise ValueError("Detuning must be strictly increasing")
    edge_points = min(40, max(0, (x.size - 7) // 4))
    if edge_points:
        x, y = x[edge_points:-edge_points], y[edge_points:-edge_points]
    y = gaussian_filter1d(y, 1.0)
    span = float(np.ptp(x))
    step = float(np.median(np.diff(x)))
    offset = float(np.percentile(y, 80))
    padding = float(np.ptp(y))
    if padding <= 0:
        return invalid
    try:
        params, covariance = curve_fit(
            negative_gaussian,
            x,
            y,
            p0=(
                offset,
                max(offset - float(y.min()), 0.1 * padding),
                float(x[np.argmin(y)]),
                max(40e3, step),
            ),
            bounds=(
                (float(y.min()) - padding, 0, float(x.min()), step / 2),
                (float(y.max()) + padding, 2 * padding, float(x.max()), span),
            ),
            maxfev=800,
        )
    except (RuntimeError, ValueError, FloatingPointError):
        return invalid
    fitted = negative_gaussian(x, *params)
    total_sum = float(np.sum((y - np.mean(y)) ** 2))
    score = (
        1 - float(np.sum((y - fitted) ** 2)) / total_sum if total_sum > 0 else np.nan
    )
    errors = np.sqrt(np.diag(covariance))
    accepted = bool(
        np.isfinite(score)
        and score >= 0.1
        and np.isfinite(errors[2])
        and errors[2] > 0
        and params[1] > 0
    )
    return {
        "center_hz": float(params[2]),
        "fwhm_hz": float(FWHM_FACTOR * abs(params[3])),
        "center_error_hz": float(errors[2]),
        "fwhm_error_hz": float(FWHM_FACTOR * errors[3]),
        "r_squared": score,
        "contrast": float(params[1]),
        "accepted": accepted,
    }
