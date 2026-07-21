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

    def test_negative_anharmonicity_places_two_photon_feature_on_left(self):
        result = simulate_qutrit_map(
            duration_us=0.1,
            detuning_mhz=np.asarray([-100.0, 100.0]),
            rabi_mhz=np.asarray([20.0]),
            t1_us=50.0,
            t_phi_us=8.0,
            anharmonicity_mhz=-200.0,
            num_steps_per_half=300,
            cutoff=None,
        )

        self.assertGreater(
            float(result.second_excited[0, 0]),
            100.0 * float(result.second_excited[0, 1]),
        )

    def test_zero_drag_matches_real_drive_path(self):
        common = {
            "duration_us": 0.4,
            "detuning_mhz": np.linspace(-1.0, 1.0, 7),
            "rabi_mhz": np.asarray([2.0, 5.0]),
            "t1_us": 50.0,
            "t_phi_us": 8.0,
            "anharmonicity_mhz": -200.0,
            "num_steps_per_half": 200,
            "cutoff": 0.02,
            "echo": True,
        }
        reference = simulate_qutrit_map(**common)
        explicit_zero = simulate_qutrit_map(**common, drag_beta=0.0)

        np.testing.assert_allclose(explicit_zero.ground, reference.ground)
        np.testing.assert_allclose(explicit_zero.excited, reference.excited)
        np.testing.assert_allclose(
            explicit_zero.second_excited, reference.second_excited
        )

    def test_drag_quadrature_is_physical_and_changes_high_drive_result(self):
        common = {
            "duration_us": 0.4,
            "detuning_mhz": np.linspace(-2.0, 2.0, 9),
            "rabi_mhz": np.asarray([20.0]),
            "t1_us": 50.0,
            "t_phi_us": 8.0,
            "anharmonicity_mhz": -200.0,
            "num_steps_per_half": 500,
            "cutoff": 0.02,
            "echo": True,
        }
        plain = simulate_qutrit_map(**common)
        drag = simulate_qutrit_map(**common, drag_beta=1.0)

        total = drag.ground + drag.excited + drag.second_excited
        np.testing.assert_allclose(total, 1.0, atol=1e-10)
        self.assertGreater(
            float(np.max(np.abs(drag.second_excited - plain.second_excited))),
            1e-8,
        )

    def test_quadratic_stark_correction_changes_driven_response(self):
        common = {
            "duration_us": 0.4,
            "detuning_mhz": np.linspace(-1.0, 1.0, 9),
            "rabi_mhz": np.asarray([20.0]),
            "t1_us": 50.0,
            "t_phi_us": 8.0,
            "anharmonicity_mhz": -200.0,
            "num_steps_per_half": 500,
            "cutoff": 0.02,
            "echo": True,
        }
        reference = simulate_qutrit_map(**common)
        corrected = simulate_qutrit_map(
            **common, stark_kappa_mhz_inv=0.0025
        )

        total = corrected.ground + corrected.excited + corrected.second_excited
        np.testing.assert_allclose(total, 1.0, atol=1e-10)
        self.assertGreater(
            float(np.max(np.abs(corrected.excited - reference.excited))),
            1e-6,
        )


if __name__ == "__main__":
    unittest.main()
