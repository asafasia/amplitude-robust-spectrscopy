"""Endpoint smoothness, product DRAG, and unchanged baseline regressions."""

from unittest.mock import patch

import numpy as np
import pytest

from echospec.simulation.edge_window import edge_window
from echospec.simulation.qutrit import simulate_qutrit_map, sine_echo_sign


@pytest.mark.parametrize("kind,ramp", [("quintic", 0.04), ("polynomial", 0)])
def test_window_endpoint_symmetry_and_derivative(kind, ramp):
    def window(t):
        return edge_window(t, 0.2, kind=kind, ramp_us=ramp)

    for t in [-0.12, -0.1, 0.1, 0.12]:
        assert window(t) == (0, 0)
    assert window(0) == (1, 0)
    for t in [0.001, 0.025, 0.07, 0.095]:
        value, derivative = window(t)
        np.testing.assert_allclose(window(-t), [value, -derivative])
        numeric = (window(t + 1e-8)[0] - window(t - 1e-8)[0]) / 2e-8
        np.testing.assert_allclose(derivative, numeric, rtol=1e-6, atol=1e-8)
    # Cubic arrival to zero: value ~ distance^3, slope ~ distance^2.
    np.testing.assert_allclose(window(.1 - 1e-7)[0] / window(.1 - 2e-7)[0],
                               1 / 8, rtol=2e-5)
    np.testing.assert_allclose(window(.1 - 1e-7)[1] / window(.1 - 2e-7)[1],
                               1 / 4, rtol=2e-5)


@pytest.mark.parametrize("kind,ramp", [("quintic", 0), ("quintic", .11),
                                       ("none", .01), ("bad", 0)])
def test_invalid_window_rejected(kind, ramp):
    with pytest.raises(ValueError):
        edge_window(0, .2, kind=kind, ramp_us=ramp)


@pytest.mark.parametrize("kind,ramp,power", [("quintic", 0.08, 3),
                                           ("polynomial", 0, 3),
                                           ("polynomial", 0, 6),
                                           ("polynomial", 0, 12)])
def test_drag_differentiates_windowed_envelope(kind, ramp, power):
    probes = {}

    def capture(state, **kwargs):
        for t in [-.1, -.09, -.04, -.006, 0, .006, .04, .09, .1]:
            probes[t] = kwargs["drive_scale"](t)
        return state, kwargs["accumulated_phase"]

    with patch("echospec.simulation.qutrit._integrate", side_effect=capture):
        simulate_qutrit_map(
            duration_us=.2, detuning_mhz=np.array([0.]), rabi_mhz=np.array([20.]),
            t1_us=51.24, t_phi_us=7.87, anharmonicity_mhz=-216.,
            num_steps_per_half=100, cutoff=.05, echo=True, drag_beta=-.5,
            echo_transition_shape="sine", echo_transition_us=.016,
            edge_window_kind=kind, edge_ramp_us=ramp, edge_window_power=power,
        )
    sigma = .1 / np.sqrt(.05**-2 - 1)

    def envelope(t):
        return ((1 + (t / sigma)**2)**-.5 * sine_echo_sign(t, .016)[0]
                * edge_window(t, .2, kind=kind, ramp_us=ramp, power=power)[0])

    for t, drive in probes.items():
        derivative = (envelope(t + 1e-8) - envelope(t - 1e-8)) / 2e-8
        q = .5 * derivative / (2 * np.pi * 216)
        np.testing.assert_allclose(drive, envelope(t) - 1j*q,
                                   atol=1e-10, rtol=1e-7)


def test_default_window_preserves_populations():
    kwargs = dict(duration_us=.2, detuning_mhz=np.array([0., .2]),
                  rabi_mhz=np.array([0., 10.]), t1_us=51.24, t_phi_us=7.87,
                  anharmonicity_mhz=-216., num_steps_per_half=500,
                  cutoff=.05, echo=True, drag_beta=-.5,
                  echo_transition_shape="sine", echo_transition_us=.016)
    reference = simulate_qutrit_map(**kwargs)
    explicit = simulate_qutrit_map(**kwargs, edge_window_kind="none")
    for key in ["ground", "excited", "second_excited"]:
        np.testing.assert_array_equal(getattr(reference, key), getattr(explicit, key))


@pytest.mark.parametrize("power", [3, 4, 6, 8, 12])
def test_polynomial_power_endpoints_and_exact_derivative(power):
    def window(t):
        return edge_window(t, 30, kind="polynomial", power=power)

    assert window(0) == (1, 0)
    for t in [-16, -15, 15, 16]:
        assert window(t) == (0, 0)
    for t in [.01, 3, 8, 14]:
        value, derivative = window(t)
        np.testing.assert_allclose(window(-t), [value, -derivative])
        numeric = (window(t + 1e-5)[0] - window(t - 1e-5)[0]) / 2e-5
        np.testing.assert_allclose(derivative, numeric, rtol=1e-6, atol=1e-10)
    # At the boundary, both slope and curvature converge to zero.
    assert abs(window(15 - 1e-6)[1]) < 1e-12
    curvature = window(15 - 1e-6)[1] / -1e-6
    assert abs(curvature) < 1e-7


@pytest.mark.parametrize("power", [2, -1, np.nan, np.inf])
def test_invalid_polynomial_power_rejected(power):
    with pytest.raises(ValueError, match="power"):
        edge_window(0, 30, kind="polynomial", power=power)


def test_power_is_not_silently_ignored():
    with pytest.raises(ValueError, match="only used"):
        edge_window(0, 30, kind="none", power=6)


def test_default_polynomial_keeps_original_cube():
    for t in np.linspace(-15, 15, 101):
        x = t / 15
        np.testing.assert_array_equal(
            edge_window(t, 30, kind="polynomial"),
            [(1 - x*x)**3, -6*x*(1 - x*x)**2 / 15],
        )
