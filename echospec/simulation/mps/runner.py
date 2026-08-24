"""Batched PyTorch CPU and Apple MPS simulation runner."""

from __future__ import annotations

import numpy as np
import torch
from tqdm.auto import tqdm

from echospec.simulation.config import SimulationConfig
from echospec.simulation.density_matrix import (
    SimulationResult,
    finalize_density_matrices,
)
from echospec.simulation.model import detuning_shift, pulse_scale

from .solver import integrate_rk4, make_superoperators, resolve_device

# Backward-compatible name used by the first version of this backend.
MpsSimulationConfig = SimulationConfig


def _finalize(rho_ri: torch.Tensor, config: SimulationConfig) -> SimulationResult:
    if rho_ri.device.type != "cpu":
        torch.mps.synchronize()
    rho_array = rho_ri.detach().to("cpu").numpy()
    rho = rho_array[..., 0] + 1j * rho_array[..., 1]
    shape = (len(config.amplitude_mhz), len(config.detuning_mhz))
    rho = rho.reshape(*shape, config.levels, config.levels)
    return finalize_density_matrices(
        rho,
        tensor_device=str(rho_ri.device),
        complex_representation="explicit float32 real/imaginary",
    )


def _run_chunk(
    config: SimulationConfig,
    *,
    detuning: torch.Tensor,
    rabi: torch.Tensor,
    device: torch.device,
    progress_bar=None,
) -> torch.Tensor:
    batch = detuning.numel()
    rho = torch.zeros(
        (batch, config.levels, config.levels, 2),
        dtype=torch.float32,
        device=device,
    )
    rho[:, 0, 0, 0] = 1.0
    half_duration = config.duration_us / 2.0
    superoperators = make_superoperators(
        levels=config.levels,
        anharmonicity=2.0 * np.pi * config.anharmonicity_mhz,
        inv_t1=1.0 / config.t1_us,
        inv_t_phi=1.0 / config.t_phi_us,
        device=device,
    )
    common = {
        "num_steps": config.num_steps_per_half,
        "detuning": detuning,
        "rabi": rabi,
        "anharmonicity": 2.0 * np.pi * config.anharmonicity_mhz,
        "inv_t1": 1.0 / config.t1_us,
        "inv_t_phi": 1.0 / config.t_phi_us,
        "superoperators": superoperators,
        "detuning_shift": lambda time: detuning_shift(config, time),
        "progress_callback": None
        if progress_bar is None
        else progress_bar.update,
        "progress_interval": max(1, config.num_steps_per_half // 500),
    }
    for time_start, time_stop, phase_sign in (
        (-half_duration, 0.0, 1.0),
        (0.0, half_duration, -1.0 if config.echo else 1.0),
    ):
        rho = integrate_rk4(
            rho,
            time_start=time_start,
            time_stop=time_stop,
            drive_scale=lambda time, sign=phase_sign: pulse_scale(
                config, time, sign
            ),
            **common,
        )
    return rho


def run_simulation(
    config: SimulationConfig, *, device: str = "mps"
) -> SimulationResult:
    """Run the full amplitude-detuning grid in one batch or explicit chunks."""
    config.validate()
    torch_device = resolve_device(device)
    detuning_grid, rabi_grid = np.meshgrid(
        2.0 * np.pi * np.asarray(config.detuning_mhz, dtype=np.float32),
        2.0 * np.pi * np.asarray(config.amplitude_mhz, dtype=np.float32),
    )
    flat_detuning = torch.as_tensor(detuning_grid.reshape(-1), device=torch_device)
    flat_rabi = torch.as_tensor(rabi_grid.reshape(-1), device=torch_device)
    if flat_detuning.device.type != torch_device.type:
        raise RuntimeError("simulation tensors were not allocated on requested device")
    chunk_size = config.chunk_size or flat_detuning.numel()
    states = []
    chunk_starts = range(0, flat_detuning.numel(), chunk_size)
    total_steps = 2 * config.num_steps_per_half * len(chunk_starts)
    description = "MPS RK4" if device == "mps" else "PyTorch CPU RK4"
    with tqdm(
        total=total_steps,
        desc=description,
        unit="step",
        disable=not config.show_progress,
    ) as progress_bar:
        for start in chunk_starts:
            stop = min(start + chunk_size, flat_detuning.numel())
            states.append(
                _run_chunk(
                    config,
                    detuning=flat_detuning[start:stop],
                    rabi=flat_rabi[start:stop],
                    device=torch_device,
                    progress_bar=progress_bar,
                )
            )
    return _finalize(torch.cat(states), config)
