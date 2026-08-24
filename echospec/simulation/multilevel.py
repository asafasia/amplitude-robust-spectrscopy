"""Vectorized finite-level Duffing-transmon spectroscopy simulations."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class MultilevelSimulationResult:
    """Final populations indexed by level, Rabi frequency, and detuning."""

    populations: NDArray[np.float64]


def _rhs(
    rho: NDArray[np.complex128],
    *,
    detuning: NDArray[np.float64],
    drive: NDArray[np.complex128],
    anharmonicity: float,
    inv_t1: float,
    inv_t_phi: float,
) -> NDArray[np.complex128]:
    levels = rho.shape[0]
    number = np.arange(levels, dtype=float)
    energies = (
        -number[:, None, None] * detuning
        + 0.5
        * anharmonicity
        * number[:, None, None]
        * (number[:, None, None] - 1.0)
    )
    commutator = (
        energies[:, None, :, :] - energies[None, :, :, :]
    ) * rho

    for lower in range(levels - 1):
        coupling = 0.5 * np.sqrt(lower + 1.0) * drive
        coupling_conjugate = np.conj(coupling)
        commutator[lower] += coupling * rho[lower + 1]
        commutator[lower + 1] += coupling_conjugate * rho[lower]
        commutator[:, lower + 1] -= rho[:, lower] * coupling
        commutator[:, lower] -= rho[:, lower + 1] * coupling_conjugate

    row = number[:, None]
    column = number[None, :]
    derivative = (
        -1j * commutator
        - 0.5 * inv_t1 * (row + column)[:, :, None, None] * rho
        - inv_t_phi * (row - column)[:, :, None, None] ** 2 * rho
    )
    jump_factor = np.sqrt(
        (number[:-1, None] + 1.0) * (number[None, :-1] + 1.0)
    )
    derivative[:-1, :-1] += (
        inv_t1 * jump_factor[:, :, None, None] * rho[1:, 1:]
    )
    return derivative


def _integrate_segment(
    rho: NDArray[np.complex128],
    *,
    time_start: float,
    time_stop: float,
    num_steps: int,
    detuning: NDArray[np.float64],
    rabi: NDArray[np.float64],
    anharmonicity: float,
    inv_t1: float,
    inv_t_phi: float,
    sigma_us: float,
    drive_sign: float,
    order: float,
) -> NDArray[np.complex128]:
    step = (time_stop - time_start) / num_steps

    def derivative(
        state: NDArray[np.complex128], time_us: float
    ) -> NDArray[np.complex128]:
        envelope = (1.0 + (time_us / sigma_us) ** 2) ** (-order)
        return _rhs(
            state,
            detuning=detuning,
            drive=drive_sign * rabi * envelope,
            anharmonicity=anharmonicity,
            inv_t1=inv_t1,
            inv_t_phi=inv_t_phi,
        )

    time_us = time_start
    for _ in range(num_steps):
        k1 = derivative(rho, time_us)
        k2 = derivative(rho + 0.5 * step * k1, time_us + 0.5 * step)
        k3 = derivative(rho + 0.5 * step * k2, time_us + 0.5 * step)
        k4 = derivative(rho + step * k3, time_us + step)
        rho += (step / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        time_us += step
    return rho


def simulate_multilevel_map(
    *,
    levels: int,
    duration_us: float,
    detuning_mhz: NDArray[np.float64],
    rabi_mhz: NDArray[np.float64],
    t1_us: float,
    t_phi_us: float,
    anharmonicity_mhz: float,
    num_steps_per_half: int,
    cutoff: float,
    echo: bool = False,
    order: float = 0.5,
) -> MultilevelSimulationResult:
    """Simulate a truncated Duffing transmon with adjacent-level damping.

    The rotating-frame Hamiltonian is

    ``H = -Delta*n + alpha*n*(n-1)/2 + Omega(t)*(a+a.dag())/2``.

    Relaxation uses ``sqrt(1/T1)*a`` and pure dephasing uses
    ``sqrt(2/T_phi)*n``, matching the repository's qutrit model. Frequencies
    are cyclic MHz and times are microseconds.
    """
    if levels < 2:
        raise ValueError("levels must be at least two")
    if duration_us <= 0.0 or min(t1_us, t_phi_us) <= 0.0:
        raise ValueError("duration, T1, and T_phi must be positive")
    if num_steps_per_half < 1:
        raise ValueError("num_steps_per_half must be positive")
    if not 0.0 < cutoff < 1.0:
        raise ValueError("cutoff must lie strictly between zero and one")
    if order <= 0.0:
        raise ValueError("order must be positive")

    detuning, rabi = np.meshgrid(
        2.0 * np.pi * np.asarray(detuning_mhz, dtype=float),
        2.0 * np.pi * np.asarray(rabi_mhz, dtype=float),
    )
    rho = np.zeros((levels, levels, *detuning.shape), dtype=np.complex128)
    rho[0, 0] = 1.0
    half_duration = duration_us / 2.0
    sigma_us = half_duration / np.sqrt(cutoff ** (-1.0 / order) - 1.0)
    common = {
        "num_steps": num_steps_per_half,
        "detuning": detuning,
        "rabi": rabi,
        "anharmonicity": 2.0 * np.pi * anharmonicity_mhz,
        "inv_t1": 1.0 / t1_us,
        "inv_t_phi": 1.0 / t_phi_us,
        "sigma_us": sigma_us,
        "order": order,
    }
    rho = _integrate_segment(
        rho,
        time_start=-half_duration,
        time_stop=0.0,
        drive_sign=1.0,
        **common,
    )
    rho = _integrate_segment(
        rho,
        time_start=0.0,
        time_stop=half_duration,
        drive_sign=-1.0 if echo else 1.0,
        **common,
    )

    populations = np.stack([np.real(rho[index, index]) for index in range(levels)])
    trace_error = float(np.max(np.abs(populations.sum(axis=0) - 1.0)))
    minimum = float(populations.min())
    maximum = float(populations.max())
    if not np.all(np.isfinite(populations)):
        raise RuntimeError("Multilevel simulation produced nonfinite populations")
    if trace_error > 2e-6:
        raise RuntimeError(f"Population trace error is {trace_error:.3g}")
    if minimum < -2e-6 or maximum > 1.0 + 2e-6:
        raise RuntimeError(
            "Simulation left the physical probability interval: "
            f"{minimum:.6g} to {maximum:.6g}"
        )
    populations = np.clip(populations, 0.0, 1.0)
    populations /= populations.sum(axis=0, keepdims=True)
    return MultilevelSimulationResult(populations=populations)
