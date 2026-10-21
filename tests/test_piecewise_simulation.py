from dataclasses import replace

import numpy as np
from scipy.integrate import solve_ivp

from echospec.simulation.config import SimulationConfig
from echospec.simulation.model import pulse_scale
from echospec.simulation.piecewise import (
    dressed_f01_center,
    lindblad_generator,
    simulate_pulse_pair,
)


def test_generator_matches_direct_lindblad_equation():
    config = SimulationConfig(levels=4)
    a = np.diag(np.sqrt(np.arange(1, 4)), 1)
    number = a.T @ a
    h = 2 * np.pi * (
        -12 * number + config.anharmonicity_mhz / 2 * number @ (number - np.eye(4))
        + 25 / 2 * (a + a.T)
    )
    state = np.array([1, 2j, -1, 0.5j])
    rho = np.outer(state, state.conj()) / np.vdot(state, state)
    expected = -1j * (h @ rho - rho @ h)
    for op in (a / np.sqrt(config.t1_us), number * np.sqrt(2 / config.t_phi_us)):
        product = op.T.conj() @ op
        expected += op @ rho @ op.T.conj() - (product @ rho + rho @ product) / 2
    actual = (lindblad_generator(config, 12, 25) @ rho.ravel()).reshape(4, 4)
    np.testing.assert_allclose(actual, expected, atol=1e-12)


def test_square_pulse_rabi_rotation_and_echo_refocusing():
    config = SimulationConfig(
        levels=2, pulse_type="square", duration_us=0.125,
        amplitude_mhz=(0.0, 2.0), detuning_mhz=(0.0,), t1_us=1e9, t2_us=1e9,
    )
    result = simulate_pulse_pair(config, segments_per_half=2)
    np.testing.assert_allclose(result["ordinary"][:, 0, 1, 1], [0, 0.5], atol=1e-9)
    np.testing.assert_allclose(result["echo"][:, 0, 1, 1], 0, atol=1e-9)


def test_shaped_pulse_agrees_with_adaptive_integration():
    config = SimulationConfig(
        levels=3, duration_us=0.08, cutoff=0.5,
        amplitude_mhz=(12.0,), detuning_mhz=(2.0,),
    )
    result = simulate_pulse_pair(config, segments_per_half=256)
    for name, signs in (("ordinary", (1, 1)), ("echo", (1, -1))):
        rho = np.zeros(9, dtype=complex)
        rho[0] = 1
        for start, sign in zip((-0.04, 0), signs, strict=True):
            def rhs(t, state, phase=sign):
                scale, _ = pulse_scale(config, t, phase)
                return lindblad_generator(config, 2.0, 12 * scale) @ state

            solution = solve_ivp(rhs, (start, start + 0.04), rho,
                                 method="DOP853", rtol=1e-10, atol=1e-12)
            assert solution.success
            rho = solution.y[:, -1]
        np.testing.assert_allclose(result[name][0, 0], rho.reshape(3, 3), atol=5e-6)


def test_long_pulse_states_are_physical_and_cutoff_converges():
    config = SimulationConfig(
        duration_us=60, cutoff=0.999, amplitude_mhz=(0.0, 100.0),
        detuning_mhz=(-125.0, 0.0, 17.0, 125.0),
    )
    coarse = simulate_pulse_pair(config, segments_per_half=64)
    fine = simulate_pulse_pair(config, segments_per_half=128)
    for name, rho in fine.items():
        np.testing.assert_allclose(np.trace(rho, axis1=-2, axis2=-1), 1, atol=1e-8)
        assert np.linalg.eigvalsh(rho).min() > -1e-9
        np.testing.assert_allclose(np.diagonal(coarse[name], axis1=-2, axis2=-1),
                                   np.diagonal(rho, axis1=-2, axis2=-1), atol=1e-4)
    square = simulate_pulse_pair(replace(config, pulse_type="square"), segments_per_half=1)
    assert np.max(abs(square["ordinary"] - fine["ordinary"])) > 1e-5


def test_dressed_center_has_correct_weak_drive_sign_and_scale():
    alpha = -217.1
    result = dressed_f01_center(2.0, alpha)
    np.testing.assert_allclose(result, -2.0**2 / (2 * alpha), rtol=2e-4)
    assert dressed_f01_center(0.0, alpha) == 0.0
