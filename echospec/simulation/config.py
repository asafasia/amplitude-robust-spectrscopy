"""Shared finite-level simulation configuration."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Literal

import numpy as np


@dataclass(frozen=True)
class SimulationConfig:
    """Backend-independent spectroscopy settings in MHz and microseconds."""

    levels: int = 4
    amplitude_mhz: tuple[float, ...] = tuple(np.linspace(0.0, 25.0, 100))
    detuning_mhz: tuple[float, ...] = tuple(np.linspace(-0.5, 0.5, 100))
    num_steps_per_half: int = 30_000
    duration_us: float = 20.0
    pulse_type: Literal["lorentzian", "gaussian", "square"] = "lorentzian"
    cutoff: float = 0.005
    width_us: float | None = None
    order: float = 0.5
    zeroed_pulse: bool = False
    echo: bool = True
    anharmonicity_mhz: float = -217.106667324065
    t1_us: float = 27.1558023040541
    t2_us: float = 6.49786215784872
    drag_beta: float = 0.0
    stark_kappa_mhz_inv: float = 0.0
    chunk_size: int | None = None
    show_progress: bool = False

    @property
    def t_phi_us(self) -> float:
        rate = 1.0 / self.t2_us - 1.0 / (2.0 * self.t1_us)
        if rate <= 0.0:
            raise ValueError("T2 must be less than 2*T1 for positive T_phi")
        return 1.0 / rate

    @property
    def sigma_us(self) -> float:
        if self.width_us is not None:
            return self.width_us
        half_duration = self.duration_us / 2.0
        if self.pulse_type == "gaussian":
            return half_duration / np.sqrt(-2.0 * np.log(self.cutoff))
        return half_duration / np.sqrt(self.cutoff ** (-1.0 / self.order) - 1.0)

    def validate(self) -> None:
        if self.levels < 2:
            raise ValueError("levels must be at least two")
        if not self.amplitude_mhz or not self.detuning_mhz:
            raise ValueError("amplitude and detuning grids must not be empty")
        if self.duration_us <= 0.0 or self.num_steps_per_half < 1:
            raise ValueError("duration and number of steps must be positive")
        if self.t1_us <= 0.0 or self.t2_us <= 0.0:
            raise ValueError("T1 and T2 must be positive")
        if not 0.0 < self.cutoff < 1.0:
            raise ValueError("cutoff must lie strictly between zero and one")
        if self.order <= 0.0:
            raise ValueError("order must be positive")
        if self.width_us is not None and self.width_us <= 0.0:
            raise ValueError("width_us must be positive")
        if self.chunk_size is not None and self.chunk_size < 1:
            raise ValueError("chunk_size must be positive")
        if self.drag_beta and self.anharmonicity_mhz == 0.0:
            raise ValueError("DRAG requires nonzero anharmonicity")
        _ = self.t_phi_us


def representative_config(config: SimulationConfig) -> SimulationConfig:
    """Return corner, center/resonant, and high-amplitude validation points."""
    amplitudes = np.asarray(config.amplitude_mhz)
    detunings = np.asarray(config.detuning_mhz)
    amp_values = tuple(
        dict.fromkeys(
            (
                float(amplitudes[0]),
                float(amplitudes[len(amplitudes) // 2]),
                float(amplitudes[-1]),
            )
        )
    )
    det_values = tuple(
        dict.fromkeys(
            (
                float(detunings[0]),
                float(detunings[np.argmin(abs(detunings))]),
                float(detunings[-1]),
            )
        )
    )
    return replace(config, amplitude_mhz=amp_values, detuning_mhz=det_values)
