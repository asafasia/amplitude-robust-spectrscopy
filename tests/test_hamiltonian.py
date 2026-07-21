import unittest

import numpy as np

from echospec.simulation.hamiltonian import Hamiltonian
from echospec.simulation.pulses import PulseType
from echospec.utils.parameters import Parameters
from ips_plots.bundle_utils import detuning_to_drive_frequency_ghz


class TestHamiltonianDetuningConvention(unittest.TestCase):
    def test_positive_drive_detuning_lowers_rotating_frame_excited_level(self):
        params = Parameters(
            detuning=2.0,
            rabi_frequency=0.0,
            anharmonicity=0.0,
            pulse_type=PulseType.SQUARE,
        )

        matrix = Hamiltonian(params).get_hamiltonian().full()

        np.testing.assert_allclose(np.diag(matrix), [0.0, -2.0])

    def test_positive_saved_detuning_increases_absolute_drive_frequency(self):
        drive_frequency_ghz = detuning_to_drive_frequency_ghz(
            detuning_hz=np.asarray([2.0e6]),
            qubit_f01_hz=5.0e9,
        )

        np.testing.assert_allclose(drive_frequency_ghz, [5.002])


if __name__ == "__main__":
    unittest.main()
