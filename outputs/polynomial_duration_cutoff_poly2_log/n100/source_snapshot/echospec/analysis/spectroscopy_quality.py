"""Dip visibility under an explicit ideal Bernoulli readout model."""

from __future__ import annotations

import numpy as np

from .dip_metrics import dip_metrics


def spectroscopy_quality(detuning_mhz, population, *, shots=1000):
    """Return width, contrast, and ideal contrast SNR for a fixed shot budget.

    Half the shots measure Pe at the dip center; a quarter measure each of
    -0.45 and +0.45 MHz. The signal is mean shoulder Pe minus center Pe.
    Independent perfect binary Pe readout is assumed. No assignment error,
    technical noise, drift, or cost of locating the center is included.
    The width retains dip_metrics' finite-window background convention.
    """
    x = np.asarray(detuning_mhz, dtype=float)
    y = np.asarray(population, dtype=float)
    if shots <= 0 or not np.isfinite(shots):
        raise ValueError("shots must be finite and positive")
    if x.size < 3 or x[0] > -0.5 or x[-1] < 0.5:
        raise ValueError("detuning must cover at least [-0.5, 0.5] MHz")
    if not np.all(np.isfinite(y)) or np.any((y < 0) | (y > 1)):
        raise ValueError("population must contain physical finite probabilities")
    result = dip_metrics(x, y)
    result.update(contrast_snr=np.nan, shoulder_contrast=np.nan,
                  contrast_per_mhz=np.nan, center_population=np.nan,
                  shoulder_population=np.nan)
    if not result["accepted"]:
        return result
    center = np.interp(result["center_hz"] / 1e6, x, y)
    left, right = np.interp([-0.45, 0.45], x, y)
    shoulder = (left + right) / 2
    contrast = shoulder - center
    variance = (2 * center * (1 - center)
                + left * (1 - left) + right * (1 - right)) / shots
    result.update(
        contrast_snr=float(contrast / np.sqrt(variance)) if variance > 0 else np.nan,
        shoulder_contrast=float(contrast),
        contrast_per_mhz=float(contrast / (result["fwhm_hz"] / 1e6)),
        center_population=float(center),
        shoulder_population=float(shoulder),
    )
    return result
