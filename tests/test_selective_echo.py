"""Independent derivative and resonant time-reversal tests for new families."""

from dataclasses import replace

import numpy as np
import pytest
from scipy.integrate import solve_ivp

from echospec.simulation.polynomial_echo import (
    PolynomialEchoPulse,
    simulate_polynomial_echo,
)
from echospec.simulation.selective_echo import (
    PhaseEchoPulse,
    SelectiveEchoPulse,
    simulate_echo_split,
)


def test_selective_tail_derivative_and_monotonicity():
    pulse = SelectiveEchoPulse()
    t = np.r_[np.linspace(.0001, .04, 31), np.linspace(.1, 14.9, 31)]
    g, dg = pulse.envelope(t)
    numeric = (pulse.envelope(t + 1e-8)[0] - pulse.envelope(t - 1e-8)[0]) / 2e-8
    np.testing.assert_allclose(dg, numeric, atol=8e-8, rtol=1e-5)
    assert np.all(dg < 0)
    assert np.all((g > 0) & (g < 1))
    assert pulse.envelope(0)[0] == pytest.approx(1)
    assert pulse.envelope(15) == (0, 0)


def test_phase_zero_matches_polynomial_and_full_complex_derivative():
    pulse = PhaseEchoPulse(phase_rad=1.2, phase_scale_us=.2)
    t = np.array([-12., -.3, -.017, -.004, 0., .003, .11, 5.])
    np.testing.assert_array_equal(replace(pulse, phase_rad=0).drive(t),
                                  PolynomialEchoPulse().drive(t))
    raw = replace(pulse, drag_beta=0)
    derivative = (raw.drive(t + 1e-8) - raw.drive(t - 1e-8)) / 2e-8
    expected = raw.drive(t) + 1j * pulse.drag_beta * derivative / (2 * np.pi * 216)
    np.testing.assert_allclose(pulse.drive(t), expected, atol=1e-9)
    np.testing.assert_allclose(raw.drive(t), -raw.drive(-t), atol=1e-14)


@pytest.mark.parametrize("amplitude", [40, 50, 60])
def test_noncommuting_phase_control_cancels_at_resonance(amplitude):
    pulse = PhaseEchoPulse(duration_us=.2, cutoff=.2, phase_rad=1.3,
                           phase_scale_us=.05, drag_beta=0)

    def rhs(t, state):
        v = np.pi * amplitude * pulse.drive(t)
        return -1j * np.array([v * state[1], v.conjugate() * state[0]])

    state = np.array([1+0j, 0j])
    joins = [-.1, -.008, 0, .008, .1]
    for start, stop in zip(joins[:-1], joins[1:], strict=True):
        solution = solve_ivp(rhs, (start, stop), state, method="DOP853",
                             rtol=1e-11, atol=1e-13, max_step=.0005)
        assert solution.success
        state = solution.y[:, -1]
    np.testing.assert_allclose(state, [1, 0], atol=2e-9)


def test_independent_cptp_solver_matches_native_qutrit():
    pulse = PhaseEchoPulse(duration_us=.2, cutoff=.2, phase_rad=.4,
                           phase_scale_us=.05)
    detuning = np.array([-.5, 0, .5])
    expected, _ = simulate_polynomial_echo(pulse, [50], detuning, dt_ns=.125)
    actual = simulate_echo_split(pulse, 50, detuning, levels=3, refinement=8)
    np.testing.assert_allclose(actual, expected[:, 0], atol=2e-5)
    np.testing.assert_allclose(actual.sum(axis=0), 1, atol=2e-11)
    assert np.min(actual) >= 0
