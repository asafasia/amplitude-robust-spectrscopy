"""Finite-delay Ramsey Fourier lines; times in us and frequencies in MHz."""

from __future__ import annotations

import numpy as np
from scipy.optimize import brentq

KINDS = ("absorption", "power", "magnitude")


def coherence_transform(offset_mhz, duration_us: float, t2_us: float):
    """Integral of exp(-t/T2) exp(-2 pi i offset t), from zero to duration.

    This is one isolated frequency component. A real cosine has two such
    components, centered at opposite frequencies. Infinity gives the exact
    untruncated limit, without sampling or padding assumptions.
    """
    if not np.isfinite(t2_us) or t2_us <= 0:
        raise ValueError("T2 must be positive and finite")
    if np.isnan(duration_us) or duration_us <= 0:
        raise ValueError("Duration must be positive")
    z = 1 / t2_us + 2j * np.pi * np.asarray(offset_mhz)
    return 1 / z if np.isinf(duration_us) else -np.expm1(-duration_us * z) / z


def cosine_transform(frequency_mhz, duration_us, t2_us, fringe_mhz, phase=0.0):
    """Continuous transform of a baseline-subtracted, unit-amplitude real trace."""
    frequency = np.asarray(frequency_mhz)
    positive = np.exp(1j * phase) * coherence_transform(
        frequency - fringe_mhz, duration_us, t2_us
    )
    negative = np.exp(-1j * phase) * coherence_transform(
        frequency + fringe_mhz, duration_us, t2_us
    )
    return (positive + negative) / 2


def normalized_spectrum(offset_mhz, duration_us, t2_us, kind="absorption"):
    """Normalize an isolated, zero-phase line to unity at its center."""
    line = coherence_transform(offset_mhz, duration_us, t2_us)
    line = line / coherence_transform(0.0, duration_us, t2_us)
    if kind == "absorption":
        return line.real
    if kind == "power":
        return np.abs(line) ** 2
    if kind == "magnitude":
        return np.abs(line)
    raise ValueError(f"Unknown spectrum kind: {kind}")


def linewidth_mhz(duration_us: float, t2_us: float, kind="absorption") -> float:
    """FWHM between the first half-height crossings of the central peak.

    Finite rectangular windows can give sidelobes; the first crossings define
    the reported central linewidth, rather than a fit to an infinite Lorentzian.
    """
    # Validate the arguments even on the infinite-duration fast path.
    normalized_spectrum(0.0, duration_us, t2_us, kind)
    if np.isinf(duration_us):
        factor = np.sqrt(3) if kind == "magnitude" else 1.0
        return float(factor / (np.pi * t2_us))
    grid = np.linspace(0, 4 * max(1 / duration_us, 1 / t2_us), 2049)
    values = normalized_spectrum(grid, duration_us, t2_us, kind) - 0.5
    crossing = np.flatnonzero(values < 0)[0]
    half_width = brentq(
        lambda frequency: normalized_spectrum(
            frequency, duration_us, t2_us, kind
        ) - 0.5,
        grid[crossing - 1], grid[crossing], xtol=1e-14,
    )
    return 2 * half_width
