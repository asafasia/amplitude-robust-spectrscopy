"""Exponential midpoint propagation for symmetric, uncorrected real pulses.

Each constant segment is propagated with the full Lindblad matrix exponential.
The segment count resolves the envelope, rather than the carrier/Rabi period;
convergence must still be checked for the requested pulse and duration.
"""

from __future__ import annotations

import numpy as np
from scipy.linalg import expm
from scipy.optimize import minimize_scalar

from .config import SimulationConfig
from .model import pulse_scale


def lindblad_generator(config, detuning_mhz, amplitude_mhz):
    """Return batched generators acting on row-major flattened density matrices."""
    n = config.levels
    eye = np.eye(n)
    lowering = np.diag(np.sqrt(np.arange(1, n)), 1)
    number = lowering.T @ lowering
    detuning, amplitude = np.broadcast_arrays(detuning_mhz, amplitude_mhz)
    h = 2 * np.pi * (
        -detuning[..., None, None] * number
        + config.anharmonicity_mhz / 2 * (number @ (number - eye))
        + amplitude[..., None, None] / 2 * (lowering + lowering.T)
    )
    left = np.einsum("...ij,kl->...ikjl", h, eye)
    right = np.einsum("ij,...kl->...ikjl", eye, h.swapaxes(-1, -2))
    generator = -1j * (left - right).reshape(*detuning.shape, n * n, n * n)
    for collapse in (
        lowering / np.sqrt(config.t1_us),
        number * np.sqrt(2 / config.t_phi_us),
    ):
        product = collapse.T.conj() @ collapse
        generator += np.kron(collapse, collapse.conj()) - 0.5 * (
            np.kron(product, eye) + np.kron(eye, product.T)
        )
    return generator


def simulate_pulse_pair(config: SimulationConfig, *, segments_per_half=32):
    """Return ordinary and echo final states as [amplitude, detuning, n, n].

    The two pulses share an even envelope and differ only by a pi phase flip
    at t=0. ``config.echo`` is ignored because both protocols are returned.
    No normalization or clipping is applied to the calculated states.
    """
    config.validate()
    if config.drag_beta or config.stark_kappa_mhz_inv:
        raise ValueError("This propagator requires zero DRAG and Stark compensation")
    if segments_per_half < 1:
        raise ValueError("segments_per_half must be positive")
    n = config.levels
    detuning, amplitude = np.meshgrid(config.detuning_mhz, config.amplitude_mhz)
    shape = detuning.shape
    detuning, amplitude = detuning.ravel(), amplitude.ravel()
    ordinary = np.empty((amplitude.size, n * n), dtype=complex)
    echo = np.empty_like(ordinary)
    parity = ((-1.0) ** (np.arange(n)[:, None] + np.arange(n))).ravel()
    dt = config.duration_us / (2 * segments_per_half)
    times = -config.duration_us / 2 + (np.arange(segments_per_half) + 0.5) * dt
    for start in range(0, amplitude.size, config.chunk_size or 256):
        stop = min(start + (config.chunk_size or 256), amplitude.size)
        rho = np.zeros((stop - start, n * n), dtype=complex)
        rho[:, 0] = 1
        propagators = []
        for t in times:
            scale, _ = pulse_scale(config, t, 1.0)
            generator = lindblad_generator(
                config, detuning[start:stop], amplitude[start:stop] * scale
            )
            propagator = expm(dt * generator)
            propagators.append(propagator)
            rho = np.einsum("bij,bj->bi", propagator, rho)
        rho_plain, rho_echo = rho.copy(), rho.copy()
        for propagator in reversed(propagators):
            rho_plain = np.einsum("bij,bj->bi", propagator, rho_plain)
            rho_echo = parity * np.einsum("bij,bj->bi", propagator, parity * rho_echo)
        ordinary[start:stop], echo[start:stop] = rho_plain, rho_echo
    return {
        "ordinary": ordinary.reshape(*shape, n, n),
        "echo": echo.reshape(*shape, n, n),
    }


def dressed_f01_center(amplitude_mhz, anharmonicity_mhz, *, levels=4):
    """Minimum upper-two dressed-state gap near the bare f01 crossing, in MHz.

    Intended for negative anharmonicity and drives up to half its magnitude.
    This Hamiltonian definition is not a maximum of final excited population.
    """
    if anharmonicity_mhz >= 0 or not 0 <= amplitude_mhz <= abs(anharmonicity_mhz) / 2:
        raise ValueError("Requires negative anharmonicity and Rabi <= |alpha|/2")
    if amplitude_mhz == 0:
        return 0.0
    number = np.arange(levels)
    drive = amplitude_mhz / 2 * np.diag(np.sqrt(np.arange(1, levels)), 1)

    def gap(detuning):
        h = np.diag(-detuning * number + anharmonicity_mhz / 2 * number * (number - 1))
        energies = np.linalg.eigvalsh(h + drive + drive.T)
        return energies[-1] - energies[-2]

    estimate = -amplitude_mhz**2 / (2 * anharmonicity_mhz)
    result = minimize_scalar(
        gap, bounds=(-2.0, 2 * estimate + 2), method="bounded",
        options={"xatol": 1e-10},
    )
    if not result.success:
        raise RuntimeError("Dressed resonance minimization failed")
    return float(result.x)
