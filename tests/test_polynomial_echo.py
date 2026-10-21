"""Waveform conventions, limiting cutoff, and independent integration checks."""

import shutil
from dataclasses import replace

import numpy as np
import pytest
from scipy.integrate import solve_ivp

from echospec.simulation.polynomial_echo import (
    PolynomialEchoPulse,
    integration_segments,
    simulate_polynomial_echo,
)
from echospec.simulation.qutrit import _rhs


@pytest.mark.parametrize("cutoff", [1e-5, 0.001, 0.4, 0.99])
def test_envelope_matches_note_formula(cutoff):
    pulse = PolynomialEchoPulse(cutoff=cutoff)
    t = np.linspace(-14.99, 14.99, 401)
    expected = ((1 + (t / pulse.sigma_us)**2)**-0.5 - cutoff) / (1 - cutoff)
    expected *= (1 - (t / 15)**2)**5
    np.testing.assert_allclose(pulse.envelope(t)[0], expected, atol=2e-14)


@pytest.mark.parametrize("cutoff", [1e-5, 0.001, 1 - 1e-12, 1.0])
def test_analytic_derivative_and_complete_drag(cutoff):
    pulse = PolynomialEchoPulse(cutoff=cutoff)
    t = np.array([-12., -.00801, -.004, -.0001, 0., .0001, .004, .00801, 12.])
    step = 1e-8
    numeric = (pulse.drive(t + step).real - pulse.drive(t - step).real) / (2 * step)
    expected_q = -pulse.drag_beta * numeric / (2 * np.pi * 216)
    np.testing.assert_allclose(-pulse.drive(t).imag, expected_q, rtol=2e-7, atol=1e-9)
    # Nonzero Q at the zero crossing proves the sign derivative was included.
    assert pulse.drive(0).imag > 0
    for edge in [-15., 15.]:
        assert pulse.envelope(edge) == (0., 0.)
        assert pulse.drive(edge) == 0


def test_cutoff_one_is_the_continuous_limit():
    pulse = PolynomialEchoPulse(cutoff=1)
    t = np.linspace(-15, 15, 501)
    np.testing.assert_allclose(pulse.envelope(t)[0], (1 - (t / 15)**2)**6)
    np.testing.assert_allclose(replace(pulse, cutoff=1 - 1e-12).drive(t),
                               pulse.drive(t), atol=2e-12)
    assert pulse.envelope(0)[0] == 1
    assert np.isinf(pulse.sigma_us)


def test_beta_zero_keeps_the_same_smoothed_i():
    pulse = PolynomialEchoPulse()
    t = np.linspace(-.01, .01, 501)
    np.testing.assert_array_equal(pulse.drive(t).real, replace(pulse, drag_beta=0).drive(t).real)
    np.testing.assert_array_equal(replace(pulse, drag_beta=0).drive(t).imag, 0)


def test_small_cutoff_grid_resolves_the_core_and_sine_joins():
    pulse = PolynomialEchoPulse(cutoff=1e-5)
    segments = integration_segments(pulse, .5)
    boundaries = [segments[0][0]] + [b for _, b, _ in segments]
    assert boundaries[0] == -15 and boundaries[-1] == 15
    assert {-0.008, 0., 0.008}.issubset(boundaries)
    central = [(a, b, n) for a, b, n in segments if a == 0]
    assert (central[0][1] - central[0][0]) / central[0][2] <= pulse.sigma_us / 16
    assert sum(n for _, _, n in integration_segments(pulse, .25)) >= 1.99 * sum(n for _, _, n in segments)


@pytest.mark.parametrize("kwargs", [{"cutoff": 0}, {"cutoff": 1.1},
                                   {"transition_us": 0}, {"duration_us": -1},
                                   {"power": .5}, {"drag_beta": np.nan}])
def test_invalid_pulse_settings_fail(kwargs):
    with pytest.raises(ValueError):
        PolynomialEchoPulse(**kwargs)


@pytest.mark.skipif(not (shutil.which("clang++") or shutil.which("g++")),
                    reason="C++ compiler unavailable")
@pytest.mark.parametrize("cutoff", [1e-5, 0.1, 1.0])
def test_native_simulation_matches_adaptive_reference(cutoff):
    pulse = PolynomialEchoPulse(duration_us=.08, cutoff=cutoff)
    amps, det = np.array([0., 60.]), np.array([-.5, .0, .5])
    population, diagnostics = simulate_polynomial_echo(pulse, amps, det, dt_ns=.25)
    d, a = np.meshgrid(2 * np.pi * det, 2 * np.pi * amps)
    state = np.zeros((6, *d.shape), dtype=complex)
    state[0] = 1

    def rhs(t, flat):
        return _rhs(flat.reshape(state.shape), detuning=d, drive=a * pulse.drive(t),
                    anharmonicity=-216 * 2 * np.pi, inv_t1=1 / 51.24,
                    inv_t_phi=1 / 7.31 - 1 / (2 * 51.24)).ravel()

    # Solve independently, retaining the feature boundaries to prevent an
    # adaptive solver from skipping a sub-ns central feature.
    for start, stop, _ in integration_segments(pulse):
        solution = solve_ivp(rhs, (start, stop), state.ravel(), method="DOP853",
                             atol=1e-11, rtol=1e-10)
        assert solution.success
        state = solution.y[:, -1].reshape(state.shape)
    np.testing.assert_allclose(population, state[:3].real, atol=3e-6, rtol=1e-4)
    assert diagnostics["max_trace_error"] < 1e-12
    np.testing.assert_allclose(population[:, 0], np.array([[1., 1., 1.],
                                                         [0., 0., 0.], [0., 0., 0.]]))


@pytest.mark.skipif(not (shutil.which("clang++") or shutil.which("g++")),
                    reason="C++ compiler unavailable")
@pytest.mark.parametrize("cutoff", [1e-5, 0.0016681005372, 1.0])
def test_default_step_converges_at_30us_and_100mhz(cutoff):
    pulse = PolynomialEchoPulse(cutoff=cutoff)
    coarse, _ = simulate_polynomial_echo(pulse, [100.], [-.5, 0., .5])
    fine, _ = simulate_polynomial_echo(pulse, [100.], [-.5, 0., .5], dt_ns=.125)
    np.testing.assert_allclose(coarse, fine, atol=2e-4, rtol=0)
