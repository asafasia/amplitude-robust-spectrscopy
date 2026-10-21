"""Finite sine-window echo regressions."""

import unittest

import numpy as np

from echospec.simulation.qutrit import simulate_qutrit_map, sine_echo_sign


def test_sine_window_has_exact_endpoints_and_smooth_slope():
    for time, sign in ((-0.012, 1), (-0.008, 1), (0, 0), (0.008, -1), (0.012, -1)):
        value, derivative = sine_echo_sign(time, 0.016)
        assert value == sign
        if time != 0:
            assert derivative == 0
    for time in (-0.006, -0.002, 0.002, 0.006):
        eps = 1e-8
        numeric = (
            sine_echo_sign(time + eps, 0.016)[0] - sine_echo_sign(time - eps, 0.016)[0]
        ) / (2 * eps)
        np.testing.assert_allclose(sine_echo_sign(time, 0.016)[1], numeric, rtol=1e-8)


def test_invalid_window_is_rejected():
    with unittest.TestCase().assertRaises(ValueError):
        sine_echo_sign(0, -1)


def test_smoothing_applies_at_zero_beta_and_zero_window_preserves_reference():
    common = dict(
        duration_us=0.2,
        detuning_mhz=np.array([0.0, 1.0]),
        rabi_mhz=np.array([20.0]),
        t1_us=51.24,
        t_phi_us=7.87,
        anharmonicity_mhz=-216.0,
        num_steps_per_half=1000,
        cutoff=0.05,
        echo=True,
        drag_beta=0,
    )
    abrupt = simulate_qutrit_map(**common)
    zero = simulate_qutrit_map(**common, echo_transition_shape="sine")
    smooth = simulate_qutrit_map(
        **common, echo_transition_shape="sine", echo_transition_us=0.016
    )
    np.testing.assert_array_equal(abrupt.excited, zero.excited)
    assert np.max(abs(smooth.excited - zero.excited)) > 1e-4
    np.testing.assert_allclose(
        smooth.ground + smooth.excited + smooth.second_excited, 1
    )


def load_tests(loader, tests, pattern):
    return unittest.TestSuite(
        unittest.FunctionTestCase(fn)
        for name, fn in globals().items()
        if name.startswith("test_")
    )


def test_sine_drag_uses_derivative_of_complete_envelope_and_absolute_alpha():
    from unittest.mock import patch

    probes = {}

    def capture(state, **kwargs):
        for t in (-0.006, -0.002, 0.0, 0.002, 0.006):
            probes[t] = kwargs["drive_scale"](t)
        return state, kwargs["accumulated_phase"]

    with patch("echospec.simulation.qutrit._integrate", side_effect=capture):
        simulate_qutrit_map(
            duration_us=0.2,
            detuning_mhz=np.array([0.0]),
            rabi_mhz=np.array([20.0]),
            t1_us=51.24,
            t_phi_us=7.87,
            anharmonicity_mhz=-216.0,
            num_steps_per_half=100,
            cutoff=0.05,
            echo=True,
            drag_beta=0.7,
            echo_transition_shape="sine",
            echo_transition_us=0.016,
        )
    sigma = 0.1 / np.sqrt(0.05**-2 - 1)

    def envelope(t):
        return (1 + (t / sigma) ** 2) ** -0.5 * sine_echo_sign(t, 0.016)[0]

    for t, drive in probes.items():
        derivative = (envelope(t + 1e-8) - envelope(t - 1e-8)) / 2e-8
        q = -0.7 * derivative / (2 * np.pi * 216)
        np.testing.assert_allclose(drive, envelope(t) - 1j * q, rtol=1e-8, atol=1e-10)
