"""Independent checks of square echoes and the comparison's local information."""

import numpy as np
from scipy.integrate import solve_ivp

from echospec.simulation.config import SimulationConfig
from echospec.simulation.rotary_echo import (
    frequency_information,
    rotary_echo_states,
    square_echo_probability,
)


def test_unitary_square_matches_closed_form_away_from_blind_spots():
    cfg = SimulationConfig(
        levels=2, pulse_type="square", duration_us=1.37,
        amplitude_mhz=(0., 0.231, 1.219, 3.417),
        detuning_mhz=(-0.37, 0., 0.013, 0.49),
    )
    states = rotary_echo_states(cfg, unitary=True)
    expected = square_echo_probability(
        np.asarray(cfg.amplitude_mhz)[:, None], cfg.detuning_mhz, cfg.duration_us
    )
    np.testing.assert_allclose(states[..., 1, 1].real, expected, atol=2e-14)
    np.testing.assert_allclose(np.trace(states, axis1=-2, axis2=-1), 1, atol=2e-14)


def test_repeated_dissipative_echo_matches_independent_bloch_equations():
    cfg = SimulationConfig(
        levels=2, pulse_type="square", duration_us=2.13,
        amplitude_mhz=(1.83,), detuning_mhz=(0.17,), t1_us=43.8, t2_us=7.6,
    )
    state = np.array([0., 0., 1.])
    cycles = 4
    for i in range(2 * cycles):
        omega = 2 * np.pi * cfg.amplitude_mhz[0] * (-1)**i
        delta = 2 * np.pi * cfg.detuning_mhz[0]

        def rhs(t, r, omega=omega, delta=delta):
            x, y, z = r
            return [-delta*y-x/cfg.t2_us, delta*x-omega*z-y/cfg.t2_us,
                    omega*y+(1-z)/cfg.t1_us]

        solution = solve_ivp(rhs, (0, cfg.duration_us / (2 * cycles)), state,
                             method="DOP853", rtol=1e-11, atol=1e-13)
        assert solution.success
        state = solution.y[:, -1]
    result = rotary_echo_states(cfg, cycles=cycles)
    np.testing.assert_allclose(result[0, 0, 1, 1].real, (1-state[2])/2, atol=1e-10)


def test_frequency_information_matches_ramsey_and_zero_for_flat_signal():
    detuning = np.linspace(-0.1, 0.1, 4001)
    interrogation_us = 2.
    p = (1 + np.sin(2 * np.pi * detuning * interrogation_us)) / 2
    result = frequency_information(detuning, p)
    np.testing.assert_allclose(result["max_fisher_per_mhz2"],
                               (2 * np.pi * interrogation_us)**2, rtol=1e-6)
    flat = frequency_information(detuning, np.full_like(detuning, .3))
    assert flat["max_fisher_per_mhz2"] < 1e-18
