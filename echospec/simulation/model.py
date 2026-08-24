"""Shared pulse, DRAG, and Stark conventions for simulation backends."""

from __future__ import annotations

import numpy as np

from .config import SimulationConfig


def pulse_scale(
    config: SimulationConfig, time_us: float, phase_sign: float
) -> tuple[float, float]:
    """Return real/imaginary drive scale in the repository convention."""
    if config.pulse_type == "square":
        base = 1.0
        derivative = 0.0
    elif config.pulse_type == "gaussian":
        scaled = time_us / config.sigma_us
        base = float(np.exp(-0.5 * scaled**2))
        derivative = -time_us / config.sigma_us**2 * base
    else:
        scaled = time_us / config.sigma_us
        base = (1.0 + scaled**2) ** (-config.order)
        derivative = (
            -2.0
            * config.order
            * time_us
            / config.sigma_us**2
            * (1.0 + scaled**2) ** (-config.order - 1.0)
        )
    if config.zeroed_pulse and config.pulse_type != "square":
        base -= config.cutoff
    alpha = 2.0 * np.pi * config.anharmonicity_mhz
    quadrature = 0.0 if not config.drag_beta else -config.drag_beta * derivative / alpha
    # Repository convention: drive = Omega_I - i*Omega_Q.
    return phase_sign * base, -phase_sign * quadrature


def detuning_shift(config: SimulationConfig, time_us: float) -> float:
    """Return the coefficient multiplying angular Rabi frequency squared."""
    base, _ = pulse_scale(config, time_us, 1.0)
    return config.stark_kappa_mhz_inv * base**2 / (2.0 * np.pi)
