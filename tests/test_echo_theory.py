"""Independent checks of the echo theory identities and pulse conventions."""

import unittest

import numpy as np

from echospec.simulation.echo_theory import (
    EchoTheoryPulse,
    bloch_population,
    population,
    propagator,
    root_curvature_times,
    rotation_integrals,
)
from echospec.simulation.polynomial_echo import PolynomialEchoPulse


class TestEchoTheory(unittest.TestCase):
    def test_polynomial_waveform_matches_existing_zero_drag_pulse(self):
        pulse = EchoTheoryPulse(30, 0.001, 5, 0.016)
        existing = PolynomialEchoPulse(drag_beta=0)
        t = np.r_[np.linspace(-15, 15, 301), np.linspace(-0.01, 0.01, 51)]
        np.testing.assert_allclose(pulse.signed_envelope(t), existing.drive(t),
                                   rtol=2e-12, atol=2e-15)

    def test_echo_half_identity_and_unitarity_for_different_shapes(self):
        for power, transition in [(None, 0), (None, 0.016), (5, 0.016)]:
            pulse = EchoTheoryPulse(10, 0.02, power, transition)
            det = np.array([-0.13, -0.01, 0, 0.01, 0.13])
            full = propagator(pulse, 8, det)
            half = propagator(pulse, 8, det, half_only=True)
            prediction = 4 * np.real(half[0, 0] * half[0, 1])**2
            np.testing.assert_allclose(abs(full[1, 0])**2, prediction, atol=2e-8)
            np.testing.assert_allclose(np.sum(abs(full[:, 0])**2, axis=0),
                                       1, atol=2e-8)
            self.assertLess(abs(full[1, 0, 2])**2, 1e-16)
            np.testing.assert_allclose(abs(full[1, 0])**2,
                                       abs(full[1, 0, ::-1])**2, atol=1e-12)

    def test_root_curvature_closed_form_matches_rotation_quadrature(self):
        pulse = EchoTheoryPulse()
        for a in [0.3, 1, 3, 10]:
            rabi = a / (2 * np.pi * pulse.tau_us)
            s, c = root_curvature_times(pulse, rabi)
            result = rotation_integrals(pulse, rabi)
            np.testing.assert_allclose([s, c], [result["S_us"], result["C_us"]],
                                       atol=5e-9, rtol=2e-9)

    def test_general_curvature_predicts_smoothed_polynomial_response(self):
        pulse = EchoTheoryPulse(30, 0.001, 5, 0.016)
        result = rotation_integrals(pulse, 30)
        delta = 1e-5
        predicted = (np.pi * delta * result["S_us"])**2
        actual = population(pulse, 30, [delta], rtol=2e-12)[0]
        np.testing.assert_allclose(actual, predicted, rtol=1e-5)

    def test_zero_rate_bloch_matches_schrodinger(self):
        pulse = EchoTheoryPulse(10, 0.02, 5, 0.016)
        det = np.array([0, 0.01, 0.1])
        np.testing.assert_allclose(bloch_population(pulse, 10, det, 0, 0),
                                   population(pulse, 10, det), atol=2e-9)

    def test_jump_floor_has_second_order_remainder(self):
        pulse = EchoTheoryPulse(1, 0.05)
        rabi = 1.3 / (2 * np.pi * pulse.tau_us)
        integral = rotation_integrals(pulse, rabi)
        errors = []
        for scale in [1, 0.5]:
            gamma1, gammaphi = scale * 1e-3, scale * 2e-3
            expected = (gamma1 * integral["relaxation_us"]
                        + gammaphi * integral["dephasing_us"])
            actual = bloch_population(pulse, rabi, [0], gamma1, gammaphi,
                                      rtol=1e-12)[0]
            errors.append(abs(actual - expected))
        self.assertAlmostEqual(errors[0] / errors[1], 4, delta=0.02)


if __name__ == "__main__":
    unittest.main()
