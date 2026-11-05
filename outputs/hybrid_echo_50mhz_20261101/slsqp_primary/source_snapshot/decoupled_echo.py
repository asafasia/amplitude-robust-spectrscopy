"""Single echo with separately controlled core, weak tails, and edge closure.

Times are microseconds. The envelope is a root-Lorentzian core multiplied by
an independently weighted tail, a generalized Lorentzian tail decay, and a
local quintic endpoint taper. Two even Gaussian phase components preserve the
odd complex control before full-waveform derivative correction. No internal
peak normalization is applied: callers must normalize the completed I-iQ drive.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class DecoupledEchoPulse:
    """Analytic pulse compatible with the existing native and split solvers.

    ``tail_gain`` changes the weak-tail weight after ``tail_join_us`` without
    changing the central envelope value. Large gains can produce envelope
    shoulders, but the unsigned envelope remains positive and adds no sign flip.
    The local taper is exactly one before the final ``edge_width_us`` at each
    endpoint, and has zero first and second derivatives at both its joins.
    """

    duration_us: float = 30.0
    core_us: float = 0.002
    tail_gain: float = 1.0
    tail_join_us: float = 0.05
    tail_scale_us: float = 4.0
    tail_power: float = 1.0
    edge_width_us: float = 4.0
    transition_us: float = 0.016
    drag_beta: float = -0.5
    anharmonicity_mhz: float = -216.0
    phase_rad: float = -1.8
    phase_scale_us: float = 0.8
    phase2_rad: float = 0.0
    phase2_scale_us: float = 4.0

    def __post_init__(self):
        if not all(np.isfinite(value) for value in vars(self).values()):
            raise ValueError("Pulse parameters must be finite")
        if self.duration_us <= 0 or not 0 < self.core_us < self.duration_us / 2:
            raise ValueError("Require positive duration and core smaller than half duration")
        if not 0 < self.edge_width_us <= self.duration_us / 2:
            raise ValueError("Edge width must lie in (0, half duration]")
        if not 0 < self.transition_us < self.duration_us:
            raise ValueError("Echo transition must lie in (0, duration)")
        if self.tail_gain <= 0 or self.tail_power < 0:
            raise ValueError("Tail gain must be positive and tail power nonnegative")
        if min(self.tail_join_us, self.tail_scale_us, self.phase_scale_us,
               self.phase2_scale_us) <= 0:
            raise ValueError("Tail and phase time scales must be positive")
        if self.anharmonicity_mhz == 0:
            raise ValueError("Anharmonicity must be nonzero")

    @property
    def sigma_us(self):
        """Core scale used by the existing adaptive-segment sampling scheme."""
        return self.core_us

    def endpoint_window(self, time_us):
        """Return the compact quintic taper and its derivative with respect to t."""
        t = np.asarray(time_us, dtype=float)
        v = np.clip((self.duration_us / 2 - abs(t)) / self.edge_width_us, 0, 1)
        window = v**3 * (10 - 15 * v + 6 * v**2)
        derivative = -30 * v**2 * (1 - v)**2 * np.sign(t) / self.edge_width_us
        return window, derivative

    def envelope(self, time_us):
        """Return g and dg/dt with g(0)=1 and smooth zero endpoints."""
        t = np.asarray(time_us, dtype=float)
        core = 1 / np.hypot(1, t / self.core_us)
        dcore = -t / self.core_us**2 * core**3
        blend = t**2 / (t**2 + self.tail_join_us**2)
        dblend = 2 * t * self.tail_join_us**2 / (t**2 + self.tail_join_us**2)**2
        weight = 1 + (self.tail_gain - 1) * blend
        dweight = (self.tail_gain - 1) * dblend
        tail_base = 1 + (t / self.tail_scale_us)**2
        tail = tail_base**(-self.tail_power / 2)
        dtail = -self.tail_power * t / self.tail_scale_us**2 / tail_base * tail
        window, dwindow = self.endpoint_window(t)
        g = core * weight * tail * window
        dg = ((dcore * weight + core * dweight) * tail * window
              + core * weight * (dtail * window + tail * dwindow))
        inside = abs(t) < self.duration_us / 2
        return np.where(inside, g, 0), np.where(inside, dg, 0)

    def phase(self, time_us):
        """Return the even phase profile and odd time derivative in rad/us."""
        t = np.asarray(time_us, dtype=float)
        first = self.phase_rad * np.exp(-(t / self.phase_scale_us)**2)
        second = self.phase2_rad * np.exp(-(t / self.phase2_scale_us)**2)
        derivative = -2 * t * (first / self.phase_scale_us**2
                               + second / self.phase2_scale_us**2)
        return first + second, derivative

    def raw_drive_and_derivative(self, time_us):
        """Odd pre-DRAG complex control and its complete analytic derivative."""
        t = np.asarray(time_us, dtype=float)
        g, dg = self.envelope(t)
        in_transition = abs(t) < self.transition_us / 2
        angle = np.pi * t / self.transition_us
        sign = np.where(in_transition, -np.sin(angle), np.where(t < 0, 1., -1.))
        dsign = np.where(in_transition, -np.pi / self.transition_us * np.cos(angle), 0)
        phase, dphase = self.phase(t)
        rotation = np.exp(1j * phase)
        drive = g * sign * rotation
        derivative = rotation * (dg * sign + g * dsign + 1j * dphase * g * sign)
        return drive, derivative

    def drive(self, time_us):
        """Return u+i*beta*du/dt/(2*pi*|alpha|), the complete I-iQ drive."""
        raw, derivative = self.raw_drive_and_derivative(time_us)
        return raw + 1j * self.drag_beta * derivative / (
            2 * np.pi * abs(self.anharmonicity_mhz))
