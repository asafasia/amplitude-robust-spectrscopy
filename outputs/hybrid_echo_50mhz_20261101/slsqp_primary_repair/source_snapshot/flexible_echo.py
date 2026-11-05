"""Smooth single echoes with positive amplitude shaping and an IQ center path.

All times are in microseconds. The raw drive is ``u=g*c*exp(1j*phase)``;
the inherited ``drive`` adds the derivative correction to this complete complex
waveform. Peak normalization remains the caller's responsibility.

At zero ``rotation_mix`` and zero odd-phase coefficients, the raw control is odd
and gives the ideal resonant two-level time-reversal echo. Nonzero mixing
deliberately breaks that symmetry: a continuous nonzero center crossing cannot
retain the automatic echo identity. Odd-phase terms allow compensation in the
wings, without restoring this identity automatically.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from typing import ClassVar

import numpy as np

from .decoupled_echo import DecoupledEchoPulse


def _gaussians(time_us, scales_us):
    """Even Gaussian basis and its time derivative, with basis on the last axis."""
    t = np.asarray(time_us, dtype=float)[..., None]
    scales = np.asarray(scales_us)
    values = np.exp(-(t / scales)**2)
    return values, -2 * t / scales**2 * values


@dataclass(frozen=True)
class FlexibleEchoPulse(DecoupledEchoPulse):
    """Decoupled envelope plus smooth amplitude, phase, and center controls.

    The positive amplitude correction is ``exp(L(t))``, where
    ``L = W(t)*sum(a_k*(exp(-(t/s_k)**2)-1))`` and ``W`` is the inherited
    endpoint window. Thus ``L(0)=L(+/-duration/2)=0`` without introducing a
    second amplitude sign reversal. Four fixed, broad scales limit the basis
    complexity; five extra even phase terms are Gaussians on fixed scales.
    Three optional odd terms ``b_k*(t/s_k)*exp(-(t/s_k)**2)`` can oppose the
    even-quadrature area introduced by a nonzero center rotation. Their effect
    must be checked dynamically; they do not enforce echo cancellation.

    Within the transition, ``x=2*t/transition_us`` and
    ``theta=pi/2*(15*x-10*x**3+3*x**5)/8``. The center factor is
    ``c=-sin(theta)+1j*rotation_mix*cos(theta)``. It joins +1 and -1 with two
    continuous derivatives. Mixing +/-1 traces a unit semicircle; mixing zero
    crosses the origin. At equal transition width its midpoint slope is 1.875
    times that of the old sine reversal, so multiplying the old width by 1.875
    gives a useful initial match.
    """

    AMP_SCALES_US: ClassVar[tuple[float, ...]] = (.1, .5, 2., 6.)
    PHASE_SCALES_US: ClassVar[tuple[float, ...]] = (.03, .12, .5, 2., 6.)
    ODD_PHASE_SCALES_US: ClassVar[tuple[float, ...]] = (.05, .2, 1.)

    rotation_mix: float = 0.0
    amp_coefficients: tuple[float, ...] = (0., 0., 0., 0.)
    phase_coefficients: tuple[float, ...] = (0., 0., 0., 0., 0.)
    odd_phase_coefficients: tuple[float, ...] = (0., 0., 0.)

    def __post_init__(self):
        # The base validator examines vars(self), which also contains our tuple
        # fields. Validate an exact base instance instead of passing it tuples.
        DecoupledEchoPulse(**{
            field.name: getattr(self, field.name) for field in fields(DecoupledEchoPulse)
        })
        if self.core_us < .002:
            raise ValueError("Flexible-echo core must be at least 0.002 us")
        if not np.isfinite(self.rotation_mix) or not -1 <= self.rotation_mix <= 1:
            raise ValueError("Center rotation mixing must be finite and in [-1, 1]")
        for name, count in (("amp_coefficients", 4), ("phase_coefficients", 5),
                            ("odd_phase_coefficients", 3)):
            values = np.asarray(getattr(self, name), dtype=float)
            if values.shape != (count,) or not np.all(np.isfinite(values)):
                raise ValueError(f"{name} must contain {count} finite coefficients")
            object.__setattr__(self, name, tuple(float(value) for value in values))

    @property
    def sigma_us(self):
        """Conservative feature scale for the native integrator's center mesh."""
        scales = [self.core_us, min(self.PHASE_SCALES_US)]
        if self.phase_rad:
            scales.append(self.phase_scale_us)
        if self.phase2_rad:
            scales.append(self.phase2_scale_us)
        return min(scales)

    def amplitude_log_correction(self, time_us):
        """Return the dimensionless log-envelope correction and its derivative."""
        basis, dbasis = _gaussians(time_us, self.AMP_SCALES_US)
        values = np.sum((basis - 1) * self.amp_coefficients, axis=-1)
        derivative = np.sum(dbasis * self.amp_coefficients, axis=-1)
        window, dwindow = self.endpoint_window(time_us)
        return window * values, dwindow * values + window * derivative

    def envelope(self, time_us):
        """Return a positive even envelope and its analytic odd derivative."""
        g, dg = super().envelope(time_us)
        correction, derivative = self.amplitude_log_correction(time_us)
        factor = np.exp(correction)
        return g * factor, (dg + g * derivative) * factor

    def phase(self, time_us):
        """Return total phase and its derivative, including even and odd terms."""
        phase, derivative = super().phase(time_us)
        basis, dbasis = _gaussians(time_us, self.PHASE_SCALES_US)
        phase = phase + np.sum(basis * self.phase_coefficients, axis=-1)
        derivative = derivative + np.sum(dbasis * self.phase_coefficients, axis=-1)
        if any(self.odd_phase_coefficients):
            odd_basis, dodd_basis = _gaussians(time_us, self.ODD_PHASE_SCALES_US)
            t = np.asarray(time_us, dtype=float)[..., None]
            scales = np.asarray(self.ODD_PHASE_SCALES_US)
            phase += np.sum(t / scales * odd_basis * self.odd_phase_coefficients,
                            axis=-1)
            derivative += np.sum((odd_basis + t * dodd_basis) / scales
                                 * self.odd_phase_coefficients, axis=-1)
        return phase, derivative

    def raw_transition(self, time_us):
        """Return the complex center path c(t) and dc/dt before phase and DRAG."""
        t = np.asarray(time_us, dtype=float)
        x = np.clip(2 * t / self.transition_us, -1., 1.)
        theta = np.pi / 16 * (15 * x - 10 * x**3 + 3 * x**5)
        dtheta = 15 * np.pi / (8 * self.transition_us) * (1 - x*x)**2
        inside = abs(t) < self.transition_us / 2
        center = -np.sin(theta) + 1j * self.rotation_mix * np.cos(theta)
        derivative = dtheta * (-np.cos(theta) - 1j * self.rotation_mix * np.sin(theta))
        return (np.where(inside, center, np.where(t < 0, 1., -1.)),
                np.where(inside, derivative, 0.))

    def raw_drive_and_derivative(self, time_us):
        """Return u and du/dt, including amplitude, center path, and phase terms."""
        g, dg = self.envelope(time_us)
        center, dcenter = self.raw_transition(time_us)
        phase, dphase = self.phase(time_us)
        rotation = np.exp(1j * phase)
        raw = g * center * rotation
        derivative = rotation * (dg * center + g * dcenter + 1j * dphase * g * center)
        return raw, derivative
