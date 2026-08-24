"""One entry point for selecting a finite-level simulation backend."""

from __future__ import annotations

from typing import Literal

from .config import SimulationConfig
from .density_matrix import SimulationResult

SolverName = Literal["qutip", "torch-cpu", "mps"]


def run_with_solver(
    config: SimulationConfig, *, solver: SolverName
) -> SimulationResult:
    """Run one configuration while changing only the selected solver."""
    if solver == "qutip":
        from .qutip.runner import run_simulation

        return run_simulation(config)
    if solver in {"torch-cpu", "mps"}:
        from .mps.runner import run_simulation

        device = "cpu" if solver == "torch-cpu" else "mps"
        return run_simulation(config, device=device)
    raise ValueError("solver must be 'qutip', 'torch-cpu', or 'mps'")
