from dataclasses import dataclass
import numpy as np
from numpy.typing import NDArray
from qutip import basis, mesolve, destroy
import qutip as qt

from echospec.utils.parameters import Parameters, N_dim, a, sx, sy, sz
from echospec.simulation.hamiltonian import Hamiltonian


@dataclass
class Options:
    num_time_points: int = 1000
    plot: bool = False
    with_fwhm: bool = False
    non_linear_sweep: bool = False
    plot_population: bool = False
    save: bool = False


class Solver:
    def __init__(self, config: Parameters, options: Options | None = None) -> None:
        self.config = config
        self.options = options or Options()

    def run(self) -> tuple[NDArray, NDArray, NDArray]:
        return self._single_run()

    def _single_run(self) -> tuple[NDArray, NDArray, NDArray]:

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
        return sx_t, sy_t, sz_t


if __name__ == "__main__":
    config = Parameters()
    solver = Solver(config)
    sx_t, sy_t, sz_t = solver.run()
    print("sx_t:", sx_t)
    print("sy_t:", sy_t)
    print("sz_t:", sz_t)
