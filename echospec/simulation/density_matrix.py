"""Shared final-state representation and physical diagnostics."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class SimulationResult:
    """Final states and derived diagnostics on an amplitude-detuning grid."""

    density_matrices: NDArray[np.complex64]
    populations: NDArray[np.float32]
    leakage: NDArray[np.float32]
    raw_trace_error: float
    trace_error: float
    hermiticity_error: float
    minimum_eigenvalue: float
    tensor_device: str
    complex_representation: str


def finalize_density_matrices(
    density_matrices: NDArray[np.complexfloating],
    *,
    tensor_device: str,
    complex_representation: str,
) -> SimulationResult:
    """Normalize trace and calculate backend-independent diagnostics."""
    rho = np.asarray(density_matrices)
    raw_traces = np.trace(rho, axis1=-2, axis2=-1)
    raw_trace_error = float(np.max(np.abs(raw_traces - 1.0)))
    rho = rho / raw_traces[..., None, None]
    populations_last = np.real(
        np.diagonal(rho, axis1=-2, axis2=-1)
    ).astype(np.float32)
    trace_error = float(
        np.max(np.abs(np.trace(rho, axis1=-2, axis2=-1) - 1.0))
    )
    hermiticity_error = float(
        np.max(np.abs(rho - np.swapaxes(rho.conj(), -1, -2)))
    )
    minimum_eigenvalue = float(np.min(np.linalg.eigvalsh(rho).real))
    grid_shape = rho.shape[:-2]
    leakage = (
        populations_last[..., 2:].sum(axis=-1)
        if rho.shape[-1] > 2
        else np.zeros(grid_shape)
    )
    return SimulationResult(
        density_matrices=rho.astype(np.complex64),
        populations=np.moveaxis(populations_last, -1, 0),
        leakage=leakage.astype(np.float32),
        raw_trace_error=raw_trace_error,
        trace_error=trace_error,
        hermiticity_error=hermiticity_error,
        minimum_eigenvalue=minimum_eigenvalue,
        tensor_device=tensor_device,
        complex_representation=complex_representation,
    )
