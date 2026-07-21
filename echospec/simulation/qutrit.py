"""Vectorized three-level transmon simulations for dense spectroscopy maps."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class QutritSimulationResult:
    """Final populations from a dense qutrit spectroscopy simulation."""

    ground: NDArray[np.float64]
    excited: NDArray[np.float64]
    second_excited: NDArray[np.float64]


def _rhs(
    state: NDArray[np.complex128],
    *,
    detuning: NDArray[np.float64],
    drive: NDArray[np.float64],
    anharmonicity: float,
    inv_t1: float,
    inv_t_phi: float,
) -> NDArray[np.complex128]:
    """Three-level Lindblad derivative in the rotating frame.

    The state stores ``rho_00, rho_11, rho_22, rho_01, rho_02, rho_12``.
    Hermiticity supplies the other three density-matrix elements.  The
    Hamiltonian is

    ``H = detuning*n + anharmonicity*n*(n-1)/2 + drive*(a+a.dag())/2``.

    Relaxation uses ``sqrt(1/T1)*a`` and pure dephasing uses
    ``sqrt(2/T_phi)*n``, matching the repository's QuTiP model.
    """
    p0, p1, p2, rho01, rho02, rho12 = state
    coupling01 = 0.5 * drive
    coupling12 = drive / np.sqrt(2.0)
    energy1 = detuning
    energy2 = 2.0 * detuning + anharmonicity

    derivative = np.empty_like(state)
    derivative[0] = -2.0 * coupling01 * rho01.imag + inv_t1 * p1
    derivative[1] = (
        2.0 * coupling01 * rho01.imag
        - 2.0 * coupling12 * rho12.imag
        - inv_t1 * p1
        + 2.0 * inv_t1 * p2
    )
    derivative[2] = 2.0 * coupling12 * rho12.imag - 2.0 * inv_t1 * p2
    derivative[3] = (
        -1j * coupling01 * (p1 - p0)
        + 1j * energy1 * rho01
        + 1j * coupling12 * rho02
        + np.sqrt(2.0) * inv_t1 * rho12
        - (0.5 * inv_t1 + inv_t_phi) * rho01
    )
    derivative[4] = (
        -1j * coupling01 * rho12
        + 1j * coupling12 * rho01
        + 1j * energy2 * rho02
        - (inv_t1 + 4.0 * inv_t_phi) * rho02
    )
    derivative[5] = (
        -1j * coupling01 * rho02
        - 1j * coupling12 * (p2 - p1)
        + 1j * (energy2 - energy1) * rho12
        - (1.5 * inv_t1 + inv_t_phi) * rho12
    )
    return derivative


def _integrate(
    state: NDArray[np.complex128],
    *,
    coordinate_start: float,
    coordinate_stop: float,
    num_steps: int,
    detuning: NDArray[np.float64],
    rabi: NDArray[np.float64],
    anharmonicity: float,
    inv_t1: float,
    inv_t_phi: float,
    drive_scale: Callable[[float], float],
    time_jacobian: Callable[[float], float],
) -> NDArray[np.complex128]:
    """Integrate one segment with vectorized fourth-order Runge--Kutta."""
    step = (coordinate_stop - coordinate_start) / num_steps

    def derivative(
        current_state: NDArray[np.complex128], coordinate: float
    ) -> NDArray[np.complex128]:
        return time_jacobian(coordinate) * _rhs(
            current_state,
            detuning=detuning,
            drive=rabi * drive_scale(coordinate),
            anharmonicity=anharmonicity,
            inv_t1=inv_t1,
            inv_t_phi=inv_t_phi,
        )

    coordinate = coordinate_start
    for _ in range(num_steps):
        k1 = derivative(state, coordinate)
        k2 = derivative(state + 0.5 * step * k1, coordinate + 0.5 * step)
        k3 = derivative(state + 0.5 * step * k2, coordinate + 0.5 * step)
        k4 = derivative(state + step * k3, coordinate + step)
        state += (step / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        coordinate += step
    return state


def _finalize(state: NDArray[np.complex128]) -> QutritSimulationResult:
    populations = np.real(state[:3])
    if not np.all(np.isfinite(populations)):
        raise RuntimeError("Qutrit simulation produced nonfinite populations")
    trace_error = float(np.max(np.abs(populations.sum(axis=0) - 1.0)))
    minimum = float(populations.min())
    maximum = float(populations.max())
    if trace_error > 2e-6:
        raise RuntimeError(f"Qutrit population trace error is {trace_error:.3g}")
    if minimum < -2e-6 or maximum > 1.0 + 2e-6:
        raise RuntimeError(
            "Qutrit simulation left the physical probability interval: "
            f"{minimum:.6g} to {maximum:.6g}"
        )
    populations = np.clip(populations, 0.0, 1.0)
    populations /= populations.sum(axis=0, keepdims=True)
    return QutritSimulationResult(*populations)


def simulate_qutrit_map(
    *,
    duration_us: float,
    detuning_mhz: NDArray[np.float64],
    rabi_mhz: NDArray[np.float64],
    t1_us: float,
    t_phi_us: float,
    anharmonicity_mhz: float,
    num_steps_per_half: int,
    cutoff: float | None = None,
    echo: bool = False,
    order: float = 0.5,
) -> QutritSimulationResult:
    """Simulate a constant or Lorentzian-derived pulse in three levels.

    Supplying ``cutoff=None`` selects a constant pulse.  Otherwise the pulse
    is a finite generalized Lorentzian; ``echo=True`` reverses its phase at
    the midpoint.  Frequencies are cyclic MHz and times are microseconds.
    """
    if duration_us <= 0:
        raise ValueError("duration_us must be positive")
    if t1_us <= 0 or t_phi_us <= 0:
        raise ValueError("t1_us and t_phi_us must be positive")
    if num_steps_per_half < 1:
        raise ValueError("num_steps_per_half must be positive")
    if cutoff is not None and not 0.0 < cutoff < 1.0:
        raise ValueError("cutoff must lie strictly between zero and one")

    detuning, rabi = np.meshgrid(
        2.0 * np.pi * np.asarray(detuning_mhz, dtype=float),
        2.0 * np.pi * np.asarray(rabi_mhz, dtype=float),
    )
    state = np.zeros((6, *detuning.shape), dtype=np.complex128)
    state[0] = 1.0
    common = {
        "num_steps": num_steps_per_half,
        "detuning": detuning,
        "rabi": rabi,
        "anharmonicity": 2.0 * np.pi * anharmonicity_mhz,
        "inv_t1": 1.0 / t1_us,
        "inv_t_phi": 1.0 / t_phi_us,
    }

    if cutoff is None:
        half_duration = duration_us / 2.0
        for drive_sign in (1.0, -1.0 if echo else 1.0):
            state = _integrate(
                state,
                coordinate_start=0.0,
                coordinate_stop=half_duration,
                drive_scale=lambda _time, sign=drive_sign: sign,
                time_jacobian=lambda _time: 1.0,
                **common,
            )
        return _finalize(state)

    sigma_us = (duration_us / 2.0) / np.sqrt(cutoff ** (-1.0 / order) - 1.0)
    half_duration = duration_us / 2.0
    for time_start, time_stop, drive_sign in (
        (-half_duration, 0.0, 1.0),
        (0.0, half_duration, -1.0 if echo else 1.0),
    ):
        state = _integrate(
            state,
            coordinate_start=time_start,
            coordinate_stop=time_stop,
            drive_scale=lambda time, sign=drive_sign: sign / (
                1.0 + (time / sigma_us) ** 2
            )
            ** order,
            time_jacobian=lambda _time: 1.0,
            **common,
        )
    return _finalize(state)
