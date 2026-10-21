"""Cutoff-subtracted polynomial echo with the corrected, full-waveform DRAG.

Units are microseconds and cyclic MHz. The endpoint c=1 is the continuous
limit of the normalized subtraction, not an uncorrected square pulse.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .qutrit_native import _kernel


@dataclass(frozen=True)
class PolynomialEchoPulse:
    duration_us: float = 30.0
    cutoff: float = 0.001
    power: float = 5.0
    transition_us: float = 0.016
    drag_beta: float = -0.5
    anharmonicity_mhz: float = -216.0

    def __post_init__(self):
        if not all(np.isfinite(v) for v in vars(self).values()):
            raise ValueError("Pulse parameters must be finite")
        if self.duration_us <= 0 or not 0 < self.cutoff <= 1:
            raise ValueError("Require duration > 0 and 0 < cutoff <= 1")
        if self.power < 1:
            raise ValueError("Polynomial power must be at least 1")
        if not 0 < self.transition_us < self.duration_us:
            raise ValueError("Require 0 < transition_us < duration_us, even at beta=0")
        if self.anharmonicity_mhz == 0:
            raise ValueError("Anharmonicity must be nonzero")

    @property
    def sigma_us(self):
        if self.cutoff == 1:
            return np.inf
        return self.duration_us / 2 * self.cutoff / np.sqrt(
            (1 - self.cutoff) * (1 + self.cutoff)
        )

    def envelope(self, time_us):
        """Return g and dg/dt without cancellation when cutoff approaches 1."""
        t = np.asarray(time_us, dtype=float)
        half, c = self.duration_us / 2, self.cutoff
        x = np.clip(t / half, -1, 1)
        y = np.maximum(0.0, 1 - x * x)
        root = np.sqrt(c * c + (1 - c) * (1 + c) * x * x)
        # Rationalized (f-c)/(1-c); at c=1 this is exactly y.
        subtracted = c * (1 + c) * y / (root * (1 + root))
        derivative = -c * (1 + c) * x / (half * root**3)
        window = y**self.power
        dwindow = -2 * self.power * x / half * y ** (self.power - 1)
        inside = np.abs(t) < half
        return (
            np.where(inside, subtracted * window, 0.0),
            np.where(inside, derivative * window + subtracted * dwindow, 0.0),
        )

    def drive(self, time_us):
        """Unit-scale I-iQ; Q=-beta*dI/dt/(2*pi*abs(alpha))."""
        t = np.asarray(time_us, dtype=float)
        g, dg = self.envelope(t)
        in_transition = np.abs(t) < self.transition_us / 2
        angle = np.pi * t / self.transition_us
        sign = np.where(in_transition, -np.sin(angle), np.where(t < 0, 1.0, -1.0))
        dsign = np.where(
            in_transition, -np.pi / self.transition_us * np.cos(angle), 0.0
        )
        i = g * sign
        q = -self.drag_beta * (dg * sign + g * dsign) / (
            2 * np.pi * abs(self.anharmonicity_mhz)
        )
        return i - 1j * q


def integration_segments(pulse, dt_ns=0.25):
    """Uniform-time segments, refined near the core and split at sine joins.

    The local step is at most sqrt(sigma^2+t^2)/16, T/64 inside the
    transition, and dt_ns everywhere. Halving dt_ns also halves the local
    feature steps, so refinement tests cover the narrowest cutoffs too.
    """
    if not np.isfinite(dt_ns) or not 0 < dt_ns <= 0.5:
        raise ValueError("dt_ns must be in (0, 0.5]; use smaller steps to refine")
    half = pulse.duration_us / 2
    radius = min(pulse.sigma_us, pulse.transition_us / 2)
    positive = [0.0, half, pulse.transition_us / 2]
    while radius < half:
        positive.append(radius)
        radius *= 2
    positive = np.unique(positive)
    boundaries = np.r_[-positive[:0:-1], positive]
    result = []
    for start, stop in zip(boundaries[:-1], boundaries[1:], strict=True):
        near = min(abs(start), abs(stop))
        local = np.hypot(pulse.sigma_us, near) / 16
        if near < pulse.transition_us / 2:
            local = min(local, pulse.transition_us / 64)
        step = min(dt_ns / 1000, local * dt_ns / 0.5)
        result.append((float(start), float(stop), int(np.ceil((stop - start) / step))))
    return result


def simulate_polynomial_echo(
    pulse, rabi_mhz, detuning_mhz, *, t1_us=51.24, t2_us=7.31, dt_ns=0.25
):
    """Return populations [level, amplitude, detuning] and raw diagnostics.

    Uses the existing native three-level Lindblad RK4 equations. There is no
    added phenomenological Stark correction, clipping, or renormalization.
    A C++17 compiler is needed on the first run to build the cached kernel.
    """
    amps, det = np.asarray(rabi_mhz, dtype=float), np.asarray(detuning_mhz, dtype=float)
    for axis in (amps, det):
        if axis.ndim != 1 or not axis.size or not np.all(np.isfinite(axis)):
            raise ValueError("Simulation axes must be nonempty, finite 1D arrays")
    if np.any(amps < 0):
        raise ValueError("Rabi amplitudes must be nonnegative")
    if not (np.isfinite(t1_us) and np.isfinite(t2_us) and 0 < t2_us <= 2 * t1_us):
        raise ValueError("Require finite T1 > 0 and 0 < T2 <= 2*T1")
    segments = integration_segments(pulse, dt_ns)
    det_grid, amp_grid = np.meshgrid(2 * np.pi * det, 2 * np.pi * amps)
    shape = det_grid.shape
    state = np.zeros((9, det_grid.size), dtype=np.float64)
    state[0] = 1
    det_flat = np.ascontiguousarray(det_grid.ravel())
    amp_flat = np.ascontiguousarray(amp_grid.ravel())
    kernel = _kernel()
    for start, stop, steps in segments:
        drive = pulse.drive(np.linspace(start, stop, 2 * steps + 1))
        kernel(
            state.ravel(), det_grid.size, steps, (stop - start) / steps,
            det_flat, amp_flat,
            np.ascontiguousarray(drive.real), np.ascontiguousarray(drive.imag),
            2 * np.pi * pulse.anharmonicity_mhz,
            1 / t1_us, 1 / t2_us - 1 / (2 * t1_us),
        )
        if not np.isfinite(state).all():
            raise RuntimeError("Nonfinite density matrix; reduce dt_ns")
    populations = state[:3].reshape(3, *shape)
    density = np.zeros((det_grid.size, 3, 3), dtype=complex)
    for j in range(3):
        density[:, j, j] = state[j]
    for j, (row, col) in enumerate(((0, 1), (0, 2), (1, 2))):
        density[:, row, col] = state[3 + 2 * j] + 1j * state[4 + 2 * j]
        density[:, col, row] = density[:, row, col].conj()
    diagnostics = dict(
        max_trace_error=float(np.max(abs(populations.sum(axis=0) - 1))),
        min_population=float(populations.min()),
        max_population=float(populations.max()),
        min_density_eigenvalue=float(np.linalg.eigvalsh(density).min()),
        num_steps=sum(s[2] for s in segments),
        min_dt_ns=min(1000 * (b - a) / n for a, b, n in segments),
        max_dt_ns=max(1000 * (b - a) / n for a, b, n in segments),
    )
    if (diagnostics["max_trace_error"] > 2e-6
            or diagnostics["min_density_eigenvalue"] < -2e-6
            or diagnostics["max_population"] > 1 + 2e-6):
        raise RuntimeError(f"Unphysical final density matrix: {diagnostics}")
    return populations, diagnostics
