"""Exact constant-segment rotary echoes, with frequencies in MHz and time in us.

The phase inversion changes the real drive sign, not the detuning. Matrix
exponentials include relaxation and dephasing; no steady-state approximation
or time-discretization of a constant segment is used.
"""

from __future__ import annotations

import numpy as np
from scipy.linalg import expm

from .config import SimulationConfig
from .piecewise import lindblad_generator


def square_echo_probability(rabi_mhz, detuning_mhz, duration_us):
    """Exact unitary two-level Pe after one square echo, broadcast over inputs."""
    omega, delta = np.broadcast_arrays(
        2 * np.pi * np.asarray(rabi_mhz), 2 * np.pi * np.asarray(detuning_mhz)
    )
    q2 = omega**2 + delta**2
    result = np.zeros_like(q2, dtype=float)
    np.divide(4 * delta**2 * omega**2, q2**2, out=result, where=q2 > 0)
    return result * np.sin(duration_us * np.sqrt(q2) / 4)**4


def rotary_echo_states(config: SimulationConfig, *, cycles=1, unitary=False):
    """Final density matrices [amplitude, detuning, level, level].

    ``cycles`` is an integer or one integer per amplitude. Every cycle has
    equal positive and negative segments. Total duration and peak amplitude
    remain fixed even when the cycle count changes. With ``unitary=True``,
    collapse terms vanish exactly.
    """
    config.validate()
    if config.drag_beta or config.stark_kappa_mhz_inv:
        raise ValueError("Square reference requires zero DRAG and Stark correction")
    if config.pulse_type != "square":
        raise ValueError("Set pulse_type='square' for this reference")
    counts = np.broadcast_to(np.asarray(cycles), (len(config.amplitude_mhz),))
    if np.any(counts < 1) or np.any(counts != np.floor(counts)):
        raise ValueError("Cycle counts must be positive integers")
    n = config.levels
    parity = ((-1.0)**(np.arange(n)[:, None] + np.arange(n))).ravel()
    initial = np.zeros(n * n, dtype=complex)
    initial[0] = 1
    result = []
    for amplitude, count in zip(config.amplitude_mhz, counts, strict=True):
        if unitary:
            # Construct the commutator directly: SimulationConfig.t_phi_us
            # intentionally rejects zero dephasing, so do not use that property.
            lower = np.diag(np.sqrt(np.arange(1, n)), 1)
            number, eye = lower.T @ lower, np.eye(n)
            h = 2 * np.pi * (
                -np.asarray(config.detuning_mhz)[:, None, None] * number
                + config.anharmonicity_mhz / 2 * number @ (number - eye)
                + amplitude / 2 * (lower + lower.T)
            )
            generator = -1j * (
                np.einsum("bij,kl->bikjl", h, eye)
                - np.einsum("ij,bkl->bikjl", eye, h.swapaxes(-1, -2))
            ).reshape(-1, n * n, n * n)
        else:
            generator = lindblad_generator(
                config, np.asarray(config.detuning_mhz), amplitude
            )
        positive = expm(generator * config.duration_us / (2 * int(count)))
        negative = parity[None, :, None] * positive * parity[None, None, :]
        cycle = negative @ positive
        complete = np.linalg.matrix_power(cycle, int(count))
        result.append((complete @ initial).reshape(-1, n, n))
    return np.asarray(result)


def frequency_information(detuning_mhz, probability, *, population_floor=1e-4):
    """Best local Bernoulli Fisher information in MHz^-2, over interior points.

    Assumes ideal binary readout and known amplitude/working point. This is
    not the cost of locating a resonance or marginalizing unknown amplitude.
    Excluding probabilities near zero/one avoids unstable finite differences
    at an exactly dark/bright point. No readout or reset overhead is included.
    """
    x, p = np.asarray(detuning_mhz), np.asarray(probability)
    if x.ndim != 1 or x.size < 5 or p.shape != x.shape:
        raise ValueError("Need matching 1D arrays with at least five detunings")
    if not np.all(np.diff(x) > 0) or not np.all(np.isfinite(p)):
        raise ValueError("Require increasing detunings and finite probabilities")
    slope = np.gradient(p, x, edge_order=2)
    valid = (p > population_floor) & (p < 1 - population_floor)
    valid[:2] = valid[-2:] = False
    information = np.zeros_like(p)
    information[valid] = slope[valid]**2 / (p[valid] * (1 - p[valid]))
    best = int(np.argmax(information))
    return {
        "max_fisher_per_mhz2": float(information[best]),
        "best_detuning_mhz": float(x[best]),
        "sigma_f_khz_sqrt_shots": (
            float(1000 / np.sqrt(information[best]))
            if information[best] > 0 else np.nan
        ),
    }
