import unittest

import numpy as np

from echospec.simulation.multilevel import simulate_multilevel_map
from echospec.simulation.qutrit import simulate_qutrit_map


class TestMultilevelSimulation(unittest.TestCase):
    def test_three_level_limit_matches_qutrit_solver(self) -> None:
        common = {
            "duration_us": 0.2,
            "detuning_mhz": np.asarray([-0.1, 0.15]),
            "rabi_mhz": np.asarray([0.0, 1.0]),
            "t1_us": 27.0,
            "t_phi_us": 7.0,
            "anharmonicity_mhz": -217.0,
            "num_steps_per_half": 200,
            "cutoff": 0.005,
            "echo": True,
            "order": 0.5,
        }
        qutrit = simulate_qutrit_map(**common)
        multilevel = simulate_multilevel_map(levels=3, **common)
        expected = np.stack(
            (qutrit.ground, qutrit.excited, qutrit.second_excited)
        )
        np.testing.assert_allclose(multilevel.populations, expected, atol=2e-12)

    def test_four_level_populations_are_normalized(self) -> None:
        result = simulate_multilevel_map(
            levels=4,
            duration_us=0.1,
            detuning_mhz=np.asarray([-0.1, 0.0, 0.1]),
            rabi_mhz=np.asarray([0.0, 2.0]),
            t1_us=27.0,
            t_phi_us=7.0,
            anharmonicity_mhz=-217.0,
            num_steps_per_half=100,
            cutoff=0.005,
            echo=True,
        )
        self.assertEqual(result.populations.shape, (4, 2, 3))
        np.testing.assert_allclose(result.populations.sum(axis=0), 1.0, atol=1e-12)
        expected_ground = np.zeros((4, 3))
        expected_ground[0] = 1.0
        np.testing.assert_allclose(
            result.populations[:, 0], expected_ground, atol=1e-12
        )


if __name__ == "__main__":
    unittest.main()
