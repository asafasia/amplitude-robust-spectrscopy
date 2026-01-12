from dataclasses import dataclass
from matplotlib import pyplot as plt
import numpy as np
from numpy.typing import NDArray
from qutip import basis, mesolve, destroy
import qutip as qt

from echospec.utils.parameters import Parameters, N_dim, a, sx, sy, sz
from echospec.simulation.hamiltonian import Hamiltonian
from echospec.results.results import ResultsSingleRun


@dataclass
class Options:
    num_time_points: int = 2000
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


if __name__ == "__main__":
    params = Parameters()
    params.eco_pulse = True
    params.cutoff = 0.0006
    options = Options(noise=0.1)
    solver = Solver(params, options)
    results = solver.run()

    print("Final z:", results.final_z)

    ts = results.time
    sz_t = results.data[2]

    plt.plot(ts, sz_t)
    plt.xlabel("Time")
    plt.ylabel("<Z>")
    plt.title("Qubit Z Expectation Value Over Time")
    plt.show()
