import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path

# When this file is executed by path, Python adds echospec/simulation to
# sys.path. That makes ``import qutip`` resolve the local backend package rather
# than the installed QuTiP library. Use the repository root as the import base.
if __package__ in {None, ""}:
    script_directory = Path(__file__).resolve().parent
    sys.path = [
        entry
        for entry in sys.path
        if Path(entry or ".").resolve() != script_directory
    ]
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import matplotlib.pyplot as plt
import numpy as np
from qutip import basis, mesolve

from echospec.results.results import ResultsSingleRun
from echospec.simulation.backends import run_with_solver
from echospec.simulation.config import SimulationConfig
from echospec.simulation.hamiltonian import Hamiltonian
from echospec.simulation.operators import N_dim, a, sx, sy, sz
from echospec.utils.parameters import Parameters


@dataclass
class Options:
    num_time_points: int = 1000
    plot: bool = False
    with_fwhm: bool = False
    non_linear_sweep: bool = False
    plot_population: bool = False
    save: bool = False
    noise: float = 0.0


class Solver:
    def __init__(self, config: Parameters, options: Options | None = None) -> None:
        self.config = config
        self.options = options or Options()

    def run(self) -> ResultsSingleRun:
        return self._single_run()

    def _single_run(self) -> ResultsSingleRun:

        tlist = np.linspace(
            -self.config.pulse_length / 2,
            self.config.pulse_length / 2,
            self.options.num_time_points,
        )

        c_ops = [
            np.sqrt(self.config.gamma_relaxation) * a,
            np.sqrt(2 * self.config.gamma_dephasing) * a.dag() * a,
        ]

        H = Hamiltonian(params=self.config).get_hamiltonian()
        psi0 = basis(N_dim, 0)

        result = mesolve(
            H,
            psi0,
            tlist,
            c_ops,
            e_ops=[sx, sy, sz],
        )

        sx_t, sy_t, sz_t = result.expect
        ts = np.array(result.times)

        single_result_raw = np.array([sx_t, sy_t, sz_t])
        results = ResultsSingleRun(
            data=single_result_raw,
            time=ts
        )

        return results


def plot_simulation_results(results, config: SimulationConfig) -> None:
    """Plot final excitation and leakage over the configured sweep."""
    amplitudes = np.asarray(config.amplitude_mhz)
    detunings = np.asarray(config.detuning_mhz)
    total_excited = results.populations[1:].sum(axis=0)

    figure, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    panels = (
        (axes[0], total_excited, "Total excited population"),
        (axes[1], results.leakage, "Leakage population"),
    )
    for axis, values, title in panels:
        image = axis.pcolormesh(
            detunings,
            amplitudes,
            values,
            shading="auto",
            vmin=0,
            vmax=0.8
        )
        axis.set_xlabel("Detuning (MHz)")
        axis.set_ylabel("Amplitude (MHz)")
        axis.set_title(title)
        figure.colorbar(image, ax=axis, label="Population")

    figure.suptitle(f"Solver: {results.tensor_device}")


def main() -> None:
    """Run one shared configuration with the selected solver."""
    solver_name = "mps"  # Change to "torch-cpu" or "qutip".

    config = SimulationConfig(
        levels=3,
        amplitude_mhz=tuple(np.linspace(0, 25, 150)),
        detuning_mhz=tuple(np.linspace(-0.5,0.5, 150)),
        duration_us=20,
        num_steps_per_half=10000,
        cutoff=0.01,
        echo=True,
        show_progress=True,
    )

    start = time.perf_counter()
    print('hi')
    results = run_with_solver(config, solver=solver_name)
    elapsed = time.perf_counter() - start
    summary = {
        "solver": solver_name,
        "device": results.tensor_device,
        "elapsed_seconds": elapsed,
        "population_shape": list(results.populations.shape),
        "populations": results.populations.tolist(),
        "maximum_leakage": float(results.leakage.max()),
        "maximum_trace_error": results.trace_error,
        "maximum_hermiticity_error": results.hermiticity_error,
        "minimum_density_matrix_eigenvalue": results.minimum_eigenvalue,
    }
    print(json.dumps(summary, indent=2))
    plot_simulation_results(results, config)
    plt.show()


if __name__ == "__main__":
    main()
