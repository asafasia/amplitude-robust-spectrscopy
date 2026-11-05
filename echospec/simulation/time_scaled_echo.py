"""Time dilation of an echo envelope and phase, with DRAG recomputed."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .flexible_echo import FlexibleEchoPulse


@dataclass(frozen=True)
class TimeScaledEchoPulse:
    """Stretch every raw-control time scale by ``factor``.

    The raw complex waveform becomes u(t/factor). Its derivative acquires
    1/factor, so DRAG is recalculated for the unchanged physical anharmonicity
    and beta. This is not a time stretch of the already-completed DRAG waveform.
    Actual peak normalization remains the caller's responsibility.
    """

    base: FlexibleEchoPulse
    factor: float

    def __post_init__(self):
        if not isinstance(self.base, FlexibleEchoPulse):
            raise TypeError("base must be a FlexibleEchoPulse")
        if not np.isfinite(self.factor) or self.factor <= 0:
            raise ValueError("factor must be finite and positive")

    @property
    def duration_us(self):
        return self.base.duration_us * self.factor

    @property
    def sigma_us(self):
        return self.base.sigma_us * self.factor

    @property
    def transition_us(self):
        return self.base.transition_us * self.factor

    @property
    def anharmonicity_mhz(self):
        return self.base.anharmonicity_mhz

    @property
    def drag_beta(self):
        return self.base.drag_beta

    def envelope(self, time_us):
        value, derivative = self.base.envelope(np.asarray(time_us) / self.factor)
        return value, derivative / self.factor

    def phase(self, time_us):
        value, derivative = self.base.phase(np.asarray(time_us) / self.factor)
        return value, derivative / self.factor

    def raw_drive_and_derivative(self, time_us):
        value, derivative = self.base.raw_drive_and_derivative(
            np.asarray(time_us) / self.factor)
        return value, derivative / self.factor

    def drive(self, time_us):
        raw, derivative = self.raw_drive_and_derivative(time_us)
        return raw + 1j * self.drag_beta * derivative / (
            2 * np.pi * abs(self.anharmonicity_mhz))
