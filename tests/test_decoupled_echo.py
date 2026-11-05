"""Independent waveform derivatives, edge continuity, and echo cancellation."""

from dataclasses import replace

import numpy as np
import pytest
from scipy.integrate import solve_ivp

from echospec.simulation.decoupled_echo import DecoupledEchoPulse
from echospec.simulation.polynomial_echo import simulate_polynomial_echo
from echospec.simulation.selective_echo import simulate_echo_split


@pytest.mark.parametrize("gain,power,edge", [(0.2, 0, 4), (1, 1.4, 15), (4, 3, 2)])
def test_envelope_derivative_and_positive_single_echo(gain, power, edge):
    pulse = DecoupledEchoPulse(tail_gain=gain, tail_power=power, edge_width_us=edge)
    t = np.r_[np.linspace(-.01, .01, 41), [-14.9, -13.5, -8., -.3, .06, 2, 12.3, 14.8]]
    g, dg = pulse.envelope(t)
    step = 1e-8
    numeric = (pulse.envelope(t + step)[0] - pulse.envelope(t - step)[0]) / (2 * step)
    np.testing.assert_allclose(dg, numeric, atol=2e-7, rtol=2e-6)
    assert np.all(g > 0)
    assert pulse.envelope(0)[0] == pytest.approx(1)
    np.testing.assert_allclose(pulse.envelope(-t)[0], g, atol=1e-14)
    np.testing.assert_allclose(pulse.envelope(-t)[1], -dg, atol=1e-14)


def test_local_taper_preserves_interior_and_closes_amplitude_and_slope():
    pulse = DecoupledEchoPulse(edge_width_us=3)
    t = np.array([-12., -10., -1., 0., 1., 10., 12.])
    window, derivative = pulse.endpoint_window(t)
    np.testing.assert_array_equal(window, 1)
    np.testing.assert_array_equal(derivative, 0)
    for edge in (-15., 15.):
        assert pulse.envelope(edge) == (0, 0)
        assert pulse.drive(edge) == 0
        # A cubic endpoint taper gives dg/dt proportional to distance squared.
        near = edge - np.sign(edge) * np.array([1e-3, 5e-4])
        values = np.abs(pulse.envelope(near)[1])
        assert values[0] / values[1] == pytest.approx(4, rel=2e-3)
    np.testing.assert_array_equal(pulse.drive([-16., 16.]), 0)


def test_full_complex_derivative_includes_phase_and_midpoint_rotation():
    pulse = DecoupledEchoPulse(phase_rad=-1.3, phase_scale_us=.2,
                               phase2_rad=.7, phase2_scale_us=.9, tail_gain=2)
    t = np.array([-14., -.6, -.04, -.0081, -.002, 0., .003, .02, .5, 12.])
    raw, derivative = pulse.raw_drive_and_derivative(t)
    step = 1e-8
    numeric = (pulse.raw_drive_and_derivative(t + step)[0]
               - pulse.raw_drive_and_derivative(t - step)[0]) / (2 * step)
    np.testing.assert_allclose(derivative, numeric, atol=2e-7, rtol=2e-6)
    np.testing.assert_allclose(pulse.raw_drive_and_derivative(-t)[0], -raw, atol=1e-14)
    expected = raw + 1j * pulse.drag_beta * numeric / (2 * np.pi * 216)
    np.testing.assert_allclose(pulse.drive(t), expected, atol=1e-9)
    np.testing.assert_array_equal(replace(pulse, drag_beta=0).drive(t), raw)


@pytest.mark.parametrize("amplitude", [5, 40, 50, 60])
def test_noncommuting_resonant_echo_cancels_for_every_amplitude(amplitude):
    pulse = DecoupledEchoPulse(duration_us=.2, core_us=.015, tail_join_us=.02,
                               tail_gain=1.5, tail_scale_us=.05, edge_width_us=.025,
                               phase_scale_us=.03, phase2_rad=.6, phase2_scale_us=.08,
                               drag_beta=0)

    def rhs(t, state):
        coupling = np.pi * amplitude * pulse.drive(t)
        return -1j * np.array([coupling * state[1], coupling.conjugate() * state[0]])

    state = np.array([1+0j, 0j])
    joins = [-.1, -.075, -.008, 0., .008, .075, .1]
    for start, stop in zip(joins[:-1], joins[1:], strict=True):
        solution = solve_ivp(rhs, (start, stop), state, method="DOP853",
                             rtol=1e-11, atol=1e-13, max_step=.0005)
        assert solution.success
        state = solution.y[:, -1]
    np.testing.assert_allclose(state, [1, 0], atol=2e-9)


def test_native_qutrit_matches_independent_cptp_propagation():
    pulse = DecoupledEchoPulse(duration_us=.2, core_us=.015, tail_join_us=.02,
                               tail_gain=1.5, tail_scale_us=.05, edge_width_us=.025,
                               phase_scale_us=.03, phase2_rad=.6, phase2_scale_us=.08)
    detuning = np.array([-.5, 0, .5])
    native, diagnostics = simulate_polynomial_echo(pulse, [50], detuning, dt_ns=.125)
    independent = simulate_echo_split(pulse, 50, detuning, levels=3, refinement=8)
    np.testing.assert_allclose(native[:, 0], independent, atol=3e-5)
    assert diagnostics["max_trace_error"] < 1e-11


@pytest.mark.parametrize("kwargs", [
    {"duration_us": 0}, {"core_us": 0}, {"core_us": 15},
    {"tail_gain": 0}, {"tail_power": -1}, {"tail_join_us": 0},
    {"tail_scale_us": 0}, {"edge_width_us": 0}, {"edge_width_us": 16},
    {"transition_us": 0}, {"transition_us": 30}, {"anharmonicity_mhz": 0},
    {"phase_scale_us": 0}, {"phase2_scale_us": 0}, {"phase_rad": np.nan},
])
def test_invalid_parameters_fail_before_propagation(kwargs):
    with pytest.raises(ValueError):
        DecoupledEchoPulse(**kwargs)
