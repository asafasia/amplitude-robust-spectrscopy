"""Analytic and independent Lindblad checks for the two-level backend."""

from dataclasses import dataclass

import numpy as np
import pytest
from scipy.integrate import solve_ivp

from echospec.simulation.bloch_echo import simulate_bloch_echo
from echospec.simulation.polynomial_echo import PolynomialEchoPulse


@dataclass
class ConstantPulse:
    duration_us: float = 0.43
    value: complex = 1.0

    def drive(self, time):
        return np.full_like(np.asarray(time), self.value, dtype=complex)


def test_constant_drive_matches_detuned_rabi_formula():
    pulse = ConstantPulse()
    amps, det = np.array([0.0, 1.3, 3.0]), np.array([-0.7, 0.0, 0.17])
    p, _ = simulate_bloch_echo(pulse, amps, det, t1_us=np.inf, t2_us=np.inf)
    a, d = np.meshgrid(amps, det, indexing="ij")
    rate = np.hypot(a, d)
    expected = (
        np.divide(a * a, rate * rate, out=np.zeros_like(a), where=rate != 0)
        * np.sin(np.pi * rate * pulse.duration_us) ** 2
    )
    np.testing.assert_allclose(p[1], expected, atol=2e-8)


def test_free_relaxation_from_excited_state():
    p, _ = simulate_bloch_echo(
        ConstantPulse(), [0.0], [0.0], t1_us=5.0, t2_us=3.0, initial_bloch=(0, 0, -1)
    )
    np.testing.assert_allclose(p[1, 0, 0], np.exp(-0.43 / 5), atol=1e-12)


def test_complex_drive_matches_independent_density_matrix_solver():
    pulse = ConstantPulse(value=0.8 + 0.3j)
    amp, det, t1, t2 = 1.3, 0.17, 5.0, 3.0
    p, _ = simulate_bloch_echo(pulse, [amp], [det], t1_us=t1, t2_us=t2)
    h = np.array(
        [
            [0, np.pi * amp * pulse.value],
            [np.pi * amp * pulse.value.conjugate(), -2 * np.pi * det],
        ]
    )
    jump = np.array([[0.0, 1.0], [0.0, 0.0]]) / np.sqrt(t1)
    phase = np.diag([1.0, -1.0]) * np.sqrt((1 / t2 - 1 / (2 * t1)) / 2)

    def rhs(t, flat):
        rho = flat.reshape(2, 2)
        value = -1j * (h @ rho - rho @ h)
        for op in (jump, phase):
            sq = op.conj().T @ op
            value += op @ rho @ op.conj().T - (sq @ rho + rho @ sq) / 2
        return value.ravel()

    result = solve_ivp(
        rhs,
        (0, pulse.duration_us),
        np.array([1, 0, 0, 0], complex),
        method="DOP853",
        rtol=1e-11,
        atol=1e-12,
    )
    assert result.success
    np.testing.assert_allclose(
        p[:, 0, 0], np.diag(result.y[:, -1].reshape(2, 2)).real, atol=1e-8
    )


def test_real_single_echo_cancels_on_resonance_without_dissipation():
    pulse = PolynomialEchoPulse(duration_us=2.0, drag_beta=0.0)
    p, _ = simulate_bloch_echo(
        pulse, [5.0, 50.0, 60.0], [0.0], t1_us=np.inf, t2_us=np.inf
    )
    np.testing.assert_allclose(p[1], 0, atol=2e-7)


@pytest.mark.parametrize("cutoff", [1e-4, 0.001, 0.1])
def test_dissipative_echo_converges_with_time_step(cutoff):
    pulse = PolynomialEchoPulse(duration_us=3.0, cutoff=cutoff, drag_beta=0.0)
    coarse, _ = simulate_bloch_echo(pulse, [5.0, 50.0], [-0.4, 0.0, 0.4])
    fine, _ = simulate_bloch_echo(pulse, [5.0, 50.0], [-0.4, 0.0, 0.4], max_dt_ns=1.0)
    np.testing.assert_allclose(coarse, fine, atol=5e-6)


def test_unphysical_inputs_are_rejected():
    with pytest.raises(ValueError):
        simulate_bloch_echo(ConstantPulse(), [1.0], [0.0], t1_us=1, t2_us=3)
    with pytest.raises(ValueError):
        simulate_bloch_echo(ConstantPulse(), [1.0], [0.0], initial_bloch=(0, 0, 1.1))


def test_initial_state_is_not_mutated_for_a_single_point():
    initial = np.array([0.0, 0.0, -1.0])
    simulate_bloch_echo(ConstantPulse(), [1.0], [0.0], initial_bloch=initial)
    np.testing.assert_array_equal(initial, [0.0, 0.0, -1.0])


def test_initial_bloch_uses_standard_pauli_y_sign():
    pulse = ConstantPulse()
    populations, _ = simulate_bloch_echo(
        pulse,
        [1.3],
        [0.0],
        t1_us=np.inf,
        t2_us=np.inf,
        initial_bloch=(0.0, 1.0, 0.0),
        max_dt_ns=1.0,
    )
    expected = (1 - np.sin(2 * np.pi * 1.3 * pulse.duration_us)) / 2
    np.testing.assert_allclose(populations[1, 0, 0], expected, atol=2e-8)
