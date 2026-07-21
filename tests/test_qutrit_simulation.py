import unittest

import numpy as np

from echospec.simulation.qutrit import simulate_qutrit_map


class TestQutritSimulation(unittest.TestCase):
    def test_zero_drive_stays_in_ground_state(self):
        result = simulate_qutrit_map(
            duration_us=1.0,
            detuning_mhz=np.asarray([-1.0, 0.0, 1.0]),
            rabi_mhz=np.asarray([0.0]),
            t1_us=50.0,
            t_phi_us=8.0,
            anharmonicity_mhz=-200.0,
            num_steps_per_half=20,
            cutoff=0.002,
            echo=True,
        )

        np.testing.assert_allclose(result.ground, 1.0, atol=1e-12)
        np.testing.assert_allclose(result.excited, 0.0, atol=1e-12)
        np.testing.assert_allclose(result.second_excited, 0.0, atol=1e-12)

    def test_populations_are_normalized_and_physical(self):
        result = simulate_qutrit_map(
            duration_us=0.2,
            detuning_mhz=np.linspace(-1.0, 1.0, 5),
            rabi_mhz=np.asarray([1.0, 3.0]),
            t1_us=50.0,
            t_phi_us=8.0,
            anharmonicity_mhz=-200.0,
            num_steps_per_half=100,
            cutoff=None,
        )

        total = result.ground + result.excited + result.second_excited
        np.testing.assert_allclose(total, 1.0, atol=1e-10)
        self.assertGreaterEqual(float(np.min(result.ground)), 0.0)
        self.assertGreaterEqual(float(np.min(result.excited)), 0.0)
        self.assertGreaterEqual(float(np.min(result.second_excited)), 0.0)
        self.assertGreater(float(np.max(result.second_excited)), 0.0)


if __name__ == "__main__":
    unittest.main()
