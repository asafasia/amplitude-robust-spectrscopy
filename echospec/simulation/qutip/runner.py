"""QuTiP runner using the shared finite-level configuration and conventions."""

from __future__ import annotations

import numpy as np
import qutip as qt
from tqdm.auto import tqdm

from echospec.simulation.config import SimulationConfig
from echospec.simulation.density_matrix import (
    SimulationResult,
    finalize_density_matrices,
)
from echospec.simulation.model import pulse_scale


def run_simulation(config: SimulationConfig) -> SimulationResult:
    """Run every configured grid point through QuTiP ``mesolve``."""
    config.validate()
    levels = config.levels
    annihilation = qt.destroy(levels)
    number = qt.num(levels)
    duffing = number * (number - qt.qeye(levels))
    collapse = [
        np.sqrt(1.0 / config.t1_us) * annihilation,
        np.sqrt(2.0 / config.t_phi_us) * number,
    ]
    half_duration = config.duration_us / 2.0
    max_step = config.duration_us / (
        2 * min(config.num_steps_per_half, 1_000)
    )
    final_states = []
    progress_bar = tqdm(
        total=len(config.amplitude_mhz) * len(config.detuning_mhz),
        desc="QuTiP parameter points",
        unit="point",
        disable=not config.show_progress,
    )
    for amplitude_mhz in config.amplitude_mhz:
        row = []
        rabi = 2.0 * np.pi * amplitude_mhz
        for detuning_mhz in config.detuning_mhz:
            base_detuning = 2.0 * np.pi * detuning_mhz
            initial_state = qt.basis(levels, 0)
            for time_start, phase_sign in (
                (-half_duration, 1.0),
                (0.0, -1.0 if config.echo else 1.0),
            ):

                def hamiltonian(
                    local_time: float,
                    _args=None,
                    start=time_start,
                    sign=phase_sign,
                    amplitude=rabi,
                    detuning=base_detuning,
                ) -> qt.Qobj:
                    pulse_time = start + local_time
                    drive_real, drive_imag = pulse_scale(
                        config, pulse_time, sign
                    )
                    drive = amplitude * complex(drive_real, drive_imag)
                    stark_shift = (
                        config.stark_kappa_mhz_inv
                        * (amplitude * drive_real) ** 2
                        / (2.0 * np.pi)
                    )
                    h0 = (
                        -(detuning + stark_shift) * number
                        + 0.5
                        * (2.0 * np.pi * config.anharmonicity_mhz)
                        * duffing
                    )
                    return h0 + 0.5 * (
                        drive * annihilation
                        + np.conj(drive) * annihilation.dag()
                    )

                result = qt.mesolve(
                    hamiltonian,
                    initial_state,
                    [0.0, half_duration],
                    collapse,
                    options={
                        "store_states": True,
                        "atol": 1e-10,
                        "rtol": 1e-8,
                        "method": "vern9",
                        "max_step": max_step,
                        "nsteps": max(
                            100_000, 8 * config.num_steps_per_half
                        ),
                    },
                )
                initial_state = result.states[-1]
            row.append(initial_state.full())
            progress_bar.update(1)
        final_states.append(row)
    progress_bar.close()
    rho = np.asarray(final_states, dtype=np.complex64)
    return finalize_density_matrices(
        rho,
        tensor_device="cpu (QuTiP)",
        complex_representation="native complex128 integration",
    )
