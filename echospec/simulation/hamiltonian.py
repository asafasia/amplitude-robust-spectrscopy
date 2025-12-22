from traitlets import Any
from echospec.utils.parameters import *
from echospec.simulation.pulses import PulseArgs, choose_pulse, PulseType
from numpy.typing import ArrayLike
import numpy as np


class Hamiltonian:
    def __init__(
        self,
        params: Parameters,
    ) -> None:
        self.params = params

    def _get_pulse_function(self):
        if self.params.pulse_type is None:
            raise ValueError("pulse_type must be set")
        return choose_pulse(
            pulse_type=self.params.pulse_type,
            eco_pulse=self.params.eco_pulse,
        )

    def get_hamiltonian(self) -> Any | list[Any]:
        qubit_Hamiltonian = (
            self.params.detuning * n
            + (self.params.anharmonicity / 2.0) * n2
        )

        drive_Hamiltonian = 0.5 * self.params.rabi_frequency * (a + a.dag())

        _pulse = self._get_pulse_function()

        def pulse(t: ArrayLike) -> np.ndarray:
            if _pulse is None:
                return np.ones_like(t)
            pulse_args = PulseArgs(
                pulse_length=self.params.pulse_length,
                cutoff=self.params.cutoff,
                order=self.params.order,
                zeroed_pulse=self.params.zeroed_pulse,
            )
            return _pulse(t, pulse_args)

        if self.params.pulse_type is PulseType.SQUARE and not self.params.eco_pulse:
            return qubit_Hamiltonian + drive_Hamiltonian

        return [qubit_Hamiltonian, [drive_Hamiltonian, pulse]]


if __name__ == "__main__":
    config = Parameters()
    hamiltonian = Hamiltonian(config)
    H = hamiltonian.get_hamiltonian()
    print(H)
