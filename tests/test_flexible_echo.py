"""Flexible waveform derivatives, smooth joins, symmetry, and propagation."""

from dataclasses import asdict, replace

import numpy as np
import pytest
from scipy.integrate import solve_ivp

from echospec.simulation.decoupled_echo import DecoupledEchoPulse
from echospec.simulation.flexible_echo import FlexibleEchoPulse
from echospec.simulation.polynomial_echo import simulate_polynomial_echo
from echospec.simulation.selective_echo import simulate_echo_split


def shaped_pulse(**kwargs):
    parameters = dict(core_us=.015, tail_gain=1.4, phase_rad=-.8,
                      phase2_rad=.3, phase2_scale_us=.6,
                      amp_coefficients=(.4, -.3, .2, -.1),
                      phase_coefficients=(.5, -.4, .3, -.2, .1))
    return FlexibleEchoPulse(**(parameters | kwargs))


@pytest.mark.parametrize("rotation_mix", [0., -.35, 1.])
@pytest.mark.parametrize("odd_coefficients", [(0., 0., 0.), (.4, -.3, .2)])
def test_complete_derivative_includes_all_controls(rotation_mix, odd_coefficients):
    pulse = shaped_pulse(rotation_mix=rotation_mix, transition_us=.03,
                         odd_phase_coefficients=odd_coefficients)
    t = np.array([-14.9, -12., -.8, -.1, -.0151, -.0149, -.004,
                  0., .003, .0149, .0151, .03, .6, 3., 14.7])
    raw, derivative = pulse.raw_drive_and_derivative(t)
    step = 1e-8
    numeric = (pulse.raw_drive_and_derivative(t + step)[0]
               - pulse.raw_drive_and_derivative(t - step)[0]) / (2 * step)
    np.testing.assert_allclose(derivative, numeric, atol=3e-7, rtol=3e-6)
    expected = raw + 1j * pulse.drag_beta * numeric / (2 * np.pi * 216)
    np.testing.assert_allclose(pulse.drive(t), expected, atol=2e-9)
    np.testing.assert_array_equal(replace(pulse, drag_beta=0).drive(t), raw)


def test_positive_even_amplitude_and_normalized_log_correction():
    pulse = shaped_pulse()
    t = np.r_[np.linspace(-.05, .05, 101), [-14.9, -13., -3., -.1, .2, 4., 14.9]]
    g, dg = pulse.envelope(t)
    step = 1e-8
    numeric = (pulse.envelope(t + step)[0] - pulse.envelope(t - step)[0]) / (2 * step)
    np.testing.assert_allclose(dg, numeric, atol=2e-7, rtol=3e-6)
    np.testing.assert_allclose(pulse.envelope(-t)[0], g, atol=1e-14)
    np.testing.assert_allclose(pulse.envelope(-t)[1], -dg, atol=1e-14)
    assert np.all(g > 0)
    assert pulse.envelope(0)[0] == 1
    correction, derivative = pulse.amplitude_log_correction([-15., 0., 15.])
    np.testing.assert_array_equal(correction, 0)
    np.testing.assert_array_equal(derivative, 0)


def test_zero_basis_coefficients_recover_base_envelope_and_phase():
    base = DecoupledEchoPulse(core_us=.015)
    pulse = FlexibleEchoPulse(**asdict(base))
    t = np.linspace(-16, 16, 1001)
    np.testing.assert_array_equal(pulse.envelope(t), base.envelope(t))
    np.testing.assert_array_equal(pulse.phase(t), base.phase(t))


@pytest.mark.parametrize("rotation_mix", [0., -.3, 1.])
def test_transition_has_c2_joins_and_expected_center(rotation_mix):
    pulse = shaped_pulse(rotation_mix=rotation_mix, transition_us=.04)
    edges = np.array([-.02, .02])
    center, derivative = pulse.raw_transition(edges)
    np.testing.assert_array_equal(center, [1., -1.])
    np.testing.assert_array_equal(derivative, 0)
    assert pulse.raw_transition(0)[0] == pytest.approx(1j * rotation_mix)
    assert pulse.raw_transition(0)[1] == pytest.approx(-1.875 * np.pi / .04)
    for edge in edges:
        # dc/dt vanishes at least quadratically, so d2c/dt2 also closes.
        h = np.array([1e-5, 5e-6])
        near = edge - np.sign(edge) * h
        slope = abs(pulse.raw_transition(near)[1])
        assert np.all(slope[1] <= .26 * slope[0])
    t = np.linspace(-.03, .03, 1001)
    norm = abs(pulse.raw_transition(t)[0])
    assert np.min(norm) == pytest.approx(abs(rotation_mix))
    assert np.max(norm) == pytest.approx(1.)
    if abs(rotation_mix) == 1:
        np.testing.assert_allclose(norm, 1., atol=3e-16)


def test_endpoint_closure_survives_all_shape_terms_and_iq_rotation():
    pulse = shaped_pulse(rotation_mix=.7, edge_width_us=3)
    t = np.array([-16., -15., 15., 16.])
    np.testing.assert_array_equal(pulse.envelope(t), 0)
    np.testing.assert_array_equal(pulse.raw_drive_and_derivative(t), 0)
    np.testing.assert_array_equal(pulse.drive(t), 0)
    for edge in (-15., 15.):
        near = edge - np.sign(edge) * np.array([1e-3, 5e-4])
        slope = abs(pulse.raw_drive_and_derivative(near)[1])
        assert slope[0] / slope[1] == pytest.approx(4., rel=3e-3)


def test_rotation_mix_deliberately_breaks_odd_raw_echo_symmetry():
    pulse = shaped_pulse(rotation_mix=0., drag_beta=0)
    t = np.linspace(-.02, .02, 101)
    raw = pulse.drive(t)
    np.testing.assert_allclose(pulse.drive(-t), -raw, atol=1e-14)
    rotating = replace(pulse, rotation_mix=.3)
    assert abs(rotating.drive(0)) == pytest.approx(.3)
    assert np.max(abs(rotating.drive(-t) + rotating.drive(t))) > .5
    odd_phase = replace(pulse, odd_phase_coefficients=(.5, -.2, .1))
    assert np.max(abs(odd_phase.drive(-t) + odd_phase.drive(t))) > .05


def test_resonant_ideal_echo_cancellation_is_lost_with_nonzero_center_rotation():
    pulse = shaped_pulse(duration_us=.2, core_us=.015, tail_join_us=.02,
                         tail_scale_us=.05, edge_width_us=.025,
                         transition_us=.03, drag_beta=0)

    def final_state(shape):
        def rhs(t, state):
            coupling = np.pi * 5 * shape.drive(t)
            return -1j * np.array([coupling * state[1], coupling.conjugate() * state[0]])

        solution = solve_ivp(rhs, (-.1, .1), [1+0j, 0j], method="DOP853",
                             rtol=1e-11, atol=1e-13, max_step=.0005)
        assert solution.success
        return solution.y[:, -1]

    np.testing.assert_allclose(final_state(pulse), [1, 0], atol=2e-9)
    assert abs(final_state(replace(pulse, rotation_mix=.5))[1])**2 > 1e-3


@pytest.mark.parametrize("rotation_mix", [0., .4])
def test_native_qutrit_matches_independent_split_solver(rotation_mix):
    pulse = shaped_pulse(duration_us=.2, core_us=.015, tail_join_us=.02,
                         tail_scale_us=.05, edge_width_us=.025,
                         transition_us=.03, rotation_mix=rotation_mix,
                         odd_phase_coefficients=(.2, -.1, .05)
                         if rotation_mix else (0., 0., 0.))
    detuning = np.array([-.5, 0., .5])
    native, diagnostics = simulate_polynomial_echo(pulse, [50], detuning, dt_ns=.125)
    independent = simulate_echo_split(pulse, 50, detuning, levels=3, refinement=8)
    np.testing.assert_allclose(native[:, 0], independent, atol=3e-5)
    assert diagnostics["max_trace_error"] < 1e-11


def test_json_lists_are_canonicalized_and_sampling_resolves_fastest_phase():
    pulse = FlexibleEchoPulse(core_us=.1, amp_coefficients=[0., 0., 0., 0.],
                               phase_coefficients=[0., 0., 0., 0., 0.])
    assert isinstance(pulse.amp_coefficients, tuple)
    assert isinstance(pulse.phase_coefficients, tuple)
    assert isinstance(pulse.odd_phase_coefficients, tuple)
    assert pulse.sigma_us == .03
    assert replace(pulse, phase_scale_us=.005).sigma_us == .005
    assert replace(pulse, core_us=.002).sigma_us == .002


@pytest.mark.parametrize("kwargs", [
    {"core_us": .0019}, {"duration_us": 0}, {"edge_width_us": 16},
    {"transition_us": 0}, {"transition_us": 30}, {"rotation_mix": 1.01},
    {"rotation_mix": np.nan}, {"amp_coefficients": [0.] * 3},
    {"phase_coefficients": [0.] * 6}, {"amp_coefficients": [[0.] * 4]},
    {"amp_coefficients": [0., 0., np.nan, 0.]},
    {"phase_coefficients": [0., 0., 0., np.inf, 0.]},
    {"odd_phase_coefficients": [0., 0.]},
    {"odd_phase_coefficients": [0., np.nan, 0.]},
])
def test_invalid_parameters_fail_before_propagation(kwargs):
    with pytest.raises(ValueError):
        FlexibleEchoPulse(**kwargs)
