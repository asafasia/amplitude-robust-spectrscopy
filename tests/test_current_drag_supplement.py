"""Check the pulse normalization and derivatives used in the new S5 scan."""

import unittest

import numpy as np

from scripts.make_current_drag_supplement import CurrentPulse


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


if __name__ == "__main__":
    unittest.main()
