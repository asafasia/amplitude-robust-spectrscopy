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

    def test_zero_echo_transition_preserves_instantaneous_echo(self):
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
            "drag_beta": 1.0,
        }
        reference = simulate_qutrit_map(**common)
        explicit_zero = simulate_qutrit_map(**common, echo_transition_us=0.0)

        for attribute in ("ground", "excited", "second_excited"):
            np.testing.assert_allclose(
                getattr(explicit_zero, attribute), getattr(reference, attribute)
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

    def test_smooth_echo_transition_supports_full_waveform_drag(self):
        common = {
            "duration_us": 0.4,
            "detuning_mhz": np.linspace(-2.0, 2.0, 9),
            "rabi_mhz": np.asarray([20.0]),
            "t1_us": 50.0,
            "t_phi_us": 8.0,
            "anharmonicity_mhz": -200.0,
            "num_steps_per_half": 1000,
            "cutoff": 0.02,
            "echo": True,
            "echo_transition_us": 0.01,
        }
        no_drag = simulate_qutrit_map(**common)
        drag = simulate_qutrit_map(**common, drag_beta=1.0)

        total = drag.ground + drag.excited + drag.second_excited
        np.testing.assert_allclose(total, 1.0, atol=1e-10)
        self.assertGreater(
            float(np.max(np.abs(drag.second_excited - no_drag.second_excited))),
            1e-8,
        )

    def test_smooth_echo_transition_requires_shaped_echo(self):
        common = {
            "duration_us": 0.4,
            "detuning_mhz": np.asarray([0.0]),
            "rabi_mhz": np.asarray([2.0]),
            "t1_us": 50.0,
            "t_phi_us": 8.0,
            "anharmonicity_mhz": -200.0,
            "num_steps_per_half": 20,
            "echo_transition_us": 0.01,
        }
        with self.assertRaisesRegex(ValueError, "shaped pulse with echo=True"):
            simulate_qutrit_map(**common, cutoff=None, echo=True)
        with self.assertRaisesRegex(ValueError, "shaped pulse with echo=True"):
            simulate_qutrit_map(**common, cutoff=0.02, echo=False)

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

    def test_accumulated_phase_matches_direct_detuning_correction(self):
        common = {
            "duration_us": 0.4,
            "detuning_mhz": np.linspace(-1.0, 1.0, 9),
            "rabi_mhz": np.asarray([5.0, 20.0]),
            "t1_us": 51.24,
            "t_phi_us": 7.871481,
            "anharmonicity_mhz": -200.0,
            "num_steps_per_half": 1000,
            "cutoff": 0.02,
            "echo": True,
            "drag_beta": 2.0,
            "stark_kappa_mhz_inv": 0.00225,
        }
        detuning = simulate_qutrit_map(
            **common, stark_correction_mode="detuning"
        )
        accumulated_phase = simulate_qutrit_map(
            **common, stark_correction_mode="accumulated_phase"
        )

        for attribute in ("ground", "excited", "second_excited"):
            np.testing.assert_allclose(
                getattr(accumulated_phase, attribute),
                getattr(detuning, attribute),
                atol=3e-9,
                rtol=0.0,
            )

    def test_instantaneous_phase_is_not_accumulated_phase(self):
        common = {
            "duration_us": 0.4,
            "detuning_mhz": np.linspace(-1.0, 1.0, 9),
            "rabi_mhz": np.asarray([20.0]),
            "t1_us": 51.24,
            "t_phi_us": 7.871481,
            "anharmonicity_mhz": -200.0,
            "num_steps_per_half": 1000,
            "cutoff": 0.02,
            "echo": True,
            "drag_beta": 2.0,
            "stark_kappa_mhz_inv": 0.00225,
        }
        accumulated = simulate_qutrit_map(
            **common, stark_correction_mode="accumulated_phase"
        )
        instantaneous = simulate_qutrit_map(
            **common, stark_correction_mode="instantaneous_phase"
        )

        self.assertGreater(
            float(np.max(np.abs(instantaneous.excited - accumulated.excited))),
            1e-4,
        )


if __name__ == "__main__":
    unittest.main()
