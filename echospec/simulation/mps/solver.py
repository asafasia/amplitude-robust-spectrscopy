"""Device handling and a batched real/imaginary Lindblad RK4 solver."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import torch


def mps_is_available() -> bool:
    """Return whether this PyTorch build can execute on Apple Metal."""
    return bool(torch.backends.mps.is_available())


def resolve_device(device: str) -> torch.device:
    """Resolve an explicitly requested CPU or MPS device."""
    if device == "cpu":
        return torch.device("cpu")
    if device != "mps":
        raise ValueError("device must be 'cpu' or 'mps'")
    if not mps_is_available():
        raise RuntimeError(
            "MPS was requested, but torch.backends.mps.is_available() is False"
        )
    return torch.device("mps")


def _complex_mul(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    """Multiply tensors whose final dimension stores real and imaginary parts."""
    real = left[..., 0] * right[..., 0] - left[..., 1] * right[..., 1]
    imag = left[..., 0] * right[..., 1] + left[..., 1] * right[..., 0]
    return torch.stack((real, imag), dim=-1)


def lindblad_rhs(
    rho: torch.Tensor,
    *,
    detuning: torch.Tensor,
    drive: torch.Tensor,
    anharmonicity: float,
    inv_t1: float,
    inv_t_phi: float,
) -> torch.Tensor:
    """Evaluate the Duffing-transmon master equation for a parameter batch.

    Complex values use an explicit float32 ``[..., real_imag]`` representation.
    This is intentional: PyTorch MPS does not provide the complex operator set
    needed by the solver, and this representation never falls back to CPU.
    """
    levels = rho.shape[1]
    number = torch.arange(levels, dtype=rho.dtype, device=rho.device)
    energies = (
        -detuning[:, None] * number[None, :]
        + 0.5
        * anharmonicity
        * number[None, :]
        * (number[None, :] - 1.0)
    )
    energy_difference = energies[:, :, None] - energies[:, None, :]
    commutator = rho * energy_difference[..., None]

    for lower in range(levels - 1):
        factor = 0.5 * (lower + 1.0) ** 0.5
        coupling = drive * factor
        coupling_conjugate = coupling * coupling.new_tensor((1.0, -1.0))
        commutator[:, lower] = (
            commutator[:, lower]
            + _complex_mul(coupling[:, None], rho[:, lower + 1])
        )
        commutator[:, lower + 1] = (
            commutator[:, lower + 1]
            + _complex_mul(coupling_conjugate[:, None], rho[:, lower])
        )
        commutator[:, :, lower + 1] = (
            commutator[:, :, lower + 1]
            - _complex_mul(rho[:, :, lower], coupling[:, None])
        )
        commutator[:, :, lower] = (
            commutator[:, :, lower]
            - _complex_mul(rho[:, :, lower + 1], coupling_conjugate[:, None])
        )

    # -i * (real + i imag) = imag - i real.
    derivative = torch.stack(
        (commutator[..., 1], -commutator[..., 0]), dim=-1
    )
    row = number[:, None]
    column = number[None, :]
    damping = (
        0.5 * inv_t1 * (row + column)
        + inv_t_phi * (row - column) ** 2
    )
    derivative = derivative - damping[None, ..., None] * rho
    jump_factor = torch.sqrt(
        (number[:-1, None] + 1.0) * (number[None, :-1] + 1.0)
    )
    derivative[:, :-1, :-1] = (
        derivative[:, :-1, :-1]
        + inv_t1 * jump_factor[None, ..., None] * rho[:, 1:, 1:]
    )
    return derivative


def make_superoperators(
    *,
    levels: int,
    anharmonicity: float,
    inv_t1: float,
    inv_t_phi: float,
    device: torch.device,
) -> torch.Tensor:
    """Build real linear maps for constant, detuning, I, and Q terms.

    The returned tensor has shape ``(4, 2*levels**2, 2*levels**2)`` and acts
    on an interleaved real/imaginary flattened density matrix. Building it is
    a one-time allocation outside the integration loop; combining the four
    maps lets every RHS evaluation use one dense batched matrix multiply.
    """
    dimension = 2 * levels * levels

    def rhs(vector: np.ndarray, detuning: float, drive: complex) -> np.ndarray:
        state = vector.reshape(levels, levels, 2)
        rho = state[..., 0] + 1j * state[..., 1]
        number = np.arange(levels, dtype=float)
        energies = (
            -number * detuning
            + 0.5 * anharmonicity * number * (number - 1.0)
        )
        commutator = (energies[:, None] - energies[None, :]) * rho
        for lower in range(levels - 1):
            coupling = 0.5 * np.sqrt(lower + 1.0) * drive
            commutator[lower] += coupling * rho[lower + 1]
            commutator[lower + 1] += np.conj(coupling) * rho[lower]
            commutator[:, lower + 1] -= rho[:, lower] * coupling
            commutator[:, lower] -= rho[:, lower + 1] * np.conj(coupling)
        row = number[:, None]
        column = number[None, :]
        derivative = (
            -1j * commutator
            - 0.5 * inv_t1 * (row + column) * rho
            - inv_t_phi * (row - column) ** 2 * rho
        )
        jump_factor = np.sqrt(
            (number[:-1, None] + 1.0) * (number[None, :-1] + 1.0)
        )
        derivative[:-1, :-1] += inv_t1 * jump_factor * rho[1:, 1:]
        output = np.empty((*derivative.shape, 2), dtype=np.float32)
        output[..., 0] = derivative.real
        output[..., 1] = derivative.imag
        return output.reshape(-1)

    def linear_map(detuning: float, drive: complex) -> np.ndarray:
        matrix = np.empty((dimension, dimension), dtype=np.float32)
        for column in range(dimension):
            basis = np.zeros(dimension, dtype=np.float32)
            basis[column] = 1.0
            matrix[:, column] = rhs(basis, detuning, drive)
        return matrix

    constant = linear_map(0.0, 0.0j)
    maps = np.stack(
        (
            constant,
            linear_map(1.0, 0.0j) - constant,
            linear_map(0.0, 1.0 + 0.0j) - constant,
            linear_map(0.0, 0.0 + 1.0j) - constant,
        )
    )
    # Make the analytically trace-preserving structure exact in float32. The
    # first diagonal derivative is dependent on the others; reconstructing its
    # superoperator rows prevents tiny per-step roundoff from accumulating over
    # the 60,000-step production integration.
    diagonal_real = [2 * (index * levels + index) for index in range(levels)]
    diagonal_imag = [index + 1 for index in diagonal_real]
    maps[:, diagonal_real[0], :] = -maps[:, diagonal_real[1:], :].sum(axis=1)
    maps[:, diagonal_imag[0], :] = -maps[:, diagonal_imag[1:], :].sum(axis=1)
    return torch.as_tensor(maps, dtype=torch.float32, device=device)


def linear_lindblad_rhs(
    state: torch.Tensor,
    *,
    detuning: torch.Tensor,
    drive_real: torch.Tensor,
    drive_imag: torch.Tensor,
    superoperators: torch.Tensor,
) -> torch.Tensor:
    """Apply all four linear Lindblad components in one batched matmul."""
    batch = state.shape[0]
    dimension = state.shape[1]
    stacked = torch.matmul(state, superoperators.reshape(4 * dimension, dimension).T)
    components = stacked.reshape(batch, 4, dimension)
    weights = torch.stack(
        (torch.ones_like(detuning), detuning, drive_real, drive_imag), dim=1
    )
    return torch.sum(components * weights[..., None], dim=1)


def integrate_rk4(
    rho: torch.Tensor,
    *,
    time_start: float,
    time_stop: float,
    num_steps: int,
    detuning: torch.Tensor,
    rabi: torch.Tensor,
    drive_scale: Callable[[float], tuple[float, float]],
    detuning_shift: Callable[[float], float],
    anharmonicity: float,
    inv_t1: float,
    inv_t_phi: float,
    superoperators: torch.Tensor | None = None,
    progress_callback: Callable[[int], None] | None = None,
    progress_interval: int = 1,
) -> torch.Tensor:
    """Integrate one smooth pulse segment with fixed-step RK4."""
    step = (time_stop - time_start) / num_steps

    def derivative(state: torch.Tensor, time_us: float) -> torch.Tensor:
        scale_real, scale_imag = drive_scale(time_us)
        instantaneous_detuning = (
            detuning + detuning_shift(time_us) * rabi.square()
        )
        if superoperators is not None:
            flat_state = state.reshape(state.shape[0], -1)
            flat_derivative = linear_lindblad_rhs(
                flat_state,
                detuning=instantaneous_detuning,
                drive_real=rabi * scale_real,
                drive_imag=rabi * scale_imag,
                superoperators=superoperators,
            )
            return flat_derivative.reshape_as(state)
        drive = torch.stack((rabi * scale_real, rabi * scale_imag), dim=-1)
        return lindblad_rhs(
            state,
            detuning=instantaneous_detuning,
            drive=drive,
            anharmonicity=anharmonicity,
            inv_t1=inv_t1,
            inv_t_phi=inv_t_phi,
        )

    time_us = time_start
    pending_progress = 0
    for step_index in range(num_steps):
        k1 = derivative(rho, time_us)
        k2 = derivative(rho + 0.5 * step * k1, time_us + 0.5 * step)
        k3 = derivative(rho + 0.5 * step * k2, time_us + 0.5 * step)
        k4 = derivative(rho + step * k3, time_us + step)
        rho = rho + (step / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        time_us += step
        pending_progress += 1
        if progress_callback is not None and (
            pending_progress >= progress_interval
            or step_index == num_steps - 1
        ):
            progress_callback(pending_progress)
            pending_progress = 0
    return rho
