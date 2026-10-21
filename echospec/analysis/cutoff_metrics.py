"""Finite-window echo-dip metrics for cutoff/duration scans."""

from __future__ import annotations

import numpy as np
from scipy.signal import find_peaks

from .spectroscopy_quality import spectroscopy_quality


def cutoff_metrics(detuning_mhz, population, *, shots=1000):
    """Report local-dip width, ideal SNR, and their explicitly unitful ratios.

    Use the established +/-0.5 MHz window even when a map spans more. Mark
    widths below six detuning intervals as unresolved. Additional minima
    with >=0.005 population prominence are counted, not silently smoothed.
    SNR uses 1/2 of shots at the center and 1/4 at each +/-0.45 MHz shoulder.
    """
    x, y = np.asarray(detuning_mhz, dtype=float), np.asarray(population, dtype=float)
    if (x.ndim != 1 or x.size < 5 or x.shape != y.shape
            or not np.isfinite(x).all() or not np.all(np.diff(x) > 0)
            or x[0] > -0.5 or x[-1] < 0.5):
        raise ValueError("Require matching increasing arrays covering +/-0.5 MHz")
    if not np.isfinite(y).all() or y.min() < -2e-6 or y.max() > 1 + 2e-6:
        raise ValueError("Population must be finite and physical")
    if not isinstance(shots, (int, np.integer)) or shots < 4 or shots % 4:
        raise ValueError("shots must be a positive integer multiple of four")
    inside = (x > -0.5) & (x < 0.5)
    xx = np.r_[-0.5, x[inside], 0.5]
    yy = np.interp(xx, x, np.clip(y, 0, 1))
    metrics = spectroscopy_quality(xx, yy, shots=shots)
    minima, _ = find_peaks(-yy, prominence=0.005)
    width = metrics["fwhm_hz"]
    snr = metrics["contrast_snr"]
    intervals = width / (1e6 * np.max(np.diff(xx)))
    status = "accepted"
    if not metrics["accepted"]:
        status = "no_resolved_dip"
    elif not np.isfinite(snr) or snr <= 0:
        status = "nonpositive_contrast"
    elif intervals < 6:
        status = "undersampled_dip"
    accepted = status == "accepted"
    metrics.update(
        accepted=accepted,
        status=status,
        prominent_minima=int(len(minima)),
        fwhm_grid_intervals=float(intervals),
        snr_per_mhz=float(snr / (width / 1e6)) if accepted else np.nan,
        fwhm_over_snr_hz=float(width / snr) if accepted else np.nan,
        fwhm_times_snr_hz=float(width * snr) if accepted else np.nan,
        shots_for_snr_10=float(shots * (10 / snr)**2) if accepted else np.nan,
    )
    return metrics
