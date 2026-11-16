"""Check the pulse normalization and derivatives used in the DRAG scan."""

import unittest

import numpy as np

from echospec.figures.drag_convention import polynomial_beta_to_paper
from scripts.make_current_drag_supplement import CurrentPulse, paper_scan_view


class CurrentDragPulseTests(unittest.TestCase):
    def test_complex_peak_normalization(self):
        t = np.linspace(-0.1, 0.1, 20001)
        for polynomial in [False, True]:
            for beta in [-1, -0.5, 0]:
                with self.subTest(polynomial=polynomial, beta=beta):
                    pulse = CurrentPulse(polynomial=polynomial, drag_beta=beta)
                    self.assertAlmostEqual(float(np.max(abs(pulse.drive(t)))), 1, places=6)

    def test_full_signed_derivative_including_midpoint(self):
        t = np.array([-0.07, -0.012, 0, 0.018, 0.1, 1, 9.5])
        h = 1e-7
        for polynomial in [False, True]:
            pulse = CurrentPulse(polynomial=polynomial)
            numeric = (pulse.drive(t + h).real - pulse.drive(t - h).real) / (2 * h)
            q_expected = -pulse.drag_beta * numeric / (2 * np.pi * abs(pulse.anharmonicity_mhz))
            np.testing.assert_allclose(-pulse.drive(t).imag, q_expected,
                                       rtol=1e-7, atol=1e-10)

    def test_polynomial_edges_vanish_and_hard_edges_remain(self):
        polynomial = CurrentPulse(polynomial=True)
        hard = CurrentPulse(polynomial=False)
        edges = [-10, 10]
        np.testing.assert_array_equal(polynomial.drive(edges), 0)
        self.assertTrue(np.all(abs(hard.drive(edges)) > 0))
        np.testing.assert_array_equal(hard.drive([-11, 11]), 0)

    def test_paper_beta_preserves_the_signed_quadrature(self):
        t = np.array([-0.07, -0.012, 0, 0.018, 0.1, 1, 9.5])
        h = 1e-7
        for polynomial in [False, True]:
            for source_beta in [-1.0, -0.5, 0.0]:
                pulse = CurrentPulse(polynomial=polynomial, drag_beta=source_beta)
                derivative = (
                    pulse.drive(t + h).real - pulse.drive(t - h).real
                ) / (2 * h)
                beta = float(polynomial_beta_to_paper(source_beta))
                alpha = 2 * np.pi * pulse.anharmonicity_mhz
                np.testing.assert_allclose(
                    -pulse.drive(t).imag, -beta * derivative / alpha,
                    rtol=1e-7, atol=1e-10,
                )
        self.assertEqual(float(polynomial_beta_to_paper(-0.5)), 0.5)

    def test_paper_scan_relabels_beta_without_reordering_results(self):
        simulation = {
            "beta": np.array([-1.0, -0.5, 0.0]),
            "hard_populations": np.arange(12).reshape(3, 2, 2),
            "hard_rms_hz": np.array([200.0, 30.0, 8000.0]),
        }
        metadata = {
            "best_sampled": {"hard": {"beta": -0.5, "rms_hz": 30.0}},
            "convergence": {"checks": [{"beta": -0.475, "rms_hz": 40.0}]},
        }
        displayed, provenance = paper_scan_view(simulation, metadata)
        np.testing.assert_array_equal(displayed["beta"], [1.0, 0.5, 0.0])
        np.testing.assert_array_equal(simulation["beta"], [-1.0, -0.5, 0.0])
        for key in ("hard_populations", "hard_rms_hz"):
            np.testing.assert_array_equal(displayed[key], simulation[key])
        self.assertEqual(provenance["best_sampled"]["hard"]["beta"], 0.5)
        self.assertEqual(provenance["best_sampled"]["hard"]["rms_hz"], 30.0)
        self.assertEqual(provenance["convergence"]["checks"][0]["beta"], 0.475)
        self.assertEqual(metadata["best_sampled"]["hard"]["beta"], -0.5)
        self.assertEqual(metadata["convergence"]["checks"][0]["beta"], -0.475)


if __name__ == "__main__":
    unittest.main()
