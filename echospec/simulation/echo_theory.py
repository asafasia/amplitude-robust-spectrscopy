"""Ideal real-drive two-level checks for the echo spectroscopy theory note.

Public pulse times are microseconds and frequencies are cyclic MHz. These
calculations deliberately exclude DRAG, Stark shifts, and transmon leakage.
"""

from dataclasses import dataclass

import numpy as np
from scipy.integrate import quad, solve_ivp


@dataclass(frozen=True)
class EchoTheoryPulse:
    duration_us: float = 20.0
    cutoff: float = 0.005
    power: int | None = None
    transition_us: float = 0.0

    def __post_init__(self):
        if not 0 < self.cutoff < 1 or self.duration_us <= 0:
            raise ValueError("Require 0 < cutoff < 1 and positive duration")
        if not 0 <= self.transition_us < self.duration_us:
            raise ValueError("Invalid midpoint transition duration")
        if self.power is not None and self.power < 0:
            raise ValueError("Polynomial power must be nonnegative")

    @property
    def tau_us(self):
        return self.duration_us * self.cutoff / (2 * np.sqrt(1 - self.cutoff**2))

    @property
    def u_limit(self):
        return np.arccosh(1 / self.cutoff)

    def envelope(self, time_us):
        t = np.asarray(time_us)
        root = 1 / np.sqrt(1 + (t / self.tau_us)**2)
        if self.power is not None:
            window = np.maximum(0, 1 - (2 * t / self.duration_us)**2)
            root = (root - self.cutoff) / (1 - self.cutoff) * window**self.power
        return np.where(abs(t) <= self.duration_us / 2, root, 0)

    def signed_envelope(self, time_us, sign_at_zero=1):
        t = np.asarray(time_us)
        sign = np.where(t < 0, 1, np.where(t > 0, -1, sign_at_zero))
        if self.transition_us > 0:
            sign = np.where(abs(t) < self.transition_us / 2,
                            -np.sin(np.pi * t / self.transition_us), sign)
        return self.envelope(t) * sign

    def boundaries(self, half_only=False):
        points = [-self.u_limit, 0, self.u_limit]
        if self.transition_us:
            edge = np.arcsinh(self.transition_us / (2 * self.tau_us))
            points += [-edge, edge]
        return sorted(p for p in points if not half_only or p <= 0)


def root_curvature_times(pulse, rabi_mhz):
    """Exact S,C in us, for an abrupt hard-cutoff root-Lorentzian."""
    if pulse.power is not None or pulse.transition_us:
        raise ValueError("Closed form applies only to the abrupt hard-cutoff pulse")
    a = 2 * np.pi * np.asarray(rabi_mhz) * pulse.tau_us
    uc = pulse.u_limit
    prefactor = 2 * pulse.tau_us / (1 + a**2)
    return (prefactor * a * (1 / pulse.cutoff - np.cos(a * uc)),
            prefactor * (np.sinh(uc) + a * np.sin(a * uc)))


def adiabatic_probability(pulse, rabi_mhz, detuning_mhz):
    """Adiabatic halves with explicit outer and midpoint jumps (no fit)."""
    if pulse.power is not None or pulse.transition_us or rabi_mhz <= 0:
        raise ValueError("Require positive drive and abrupt hard-cutoff pulse")
    a = 2 * np.pi * rabi_mhz * pulse.tau_us
    delta = np.atleast_1d(detuning_mhz) / rabi_mhz
    phase = np.array([
        a * quad(lambda u, d=d: np.sqrt(1 + (d * np.cosh(u))**2),
                 0, pulse.u_limit, epsabs=1e-10)[0]
        for d in delta
    ])
    c = pulse.cutoff
    return delta**2 * (1 - c * np.cos(phase))**2 / (
        (delta**2 + c**2) * (delta**2 + 1)
    )


def propagator(pulse, rabi_mhz, detuning_mhz, *, half_only=False, rtol=2e-10):
    """Independent full/half Schrödinger propagation; shape (2, 2, detuning)."""
    detuning = 2 * np.pi * np.atleast_1d(detuning_mhz)
    state = np.repeat(np.eye(2, dtype=complex)[:, :, None], detuning.size, axis=2)
    bounds = pulse.boundaries(half_only)
    for start, stop in zip(bounds[:-1], bounds[1:], strict=True):
        sign = 1 if start < 0 else -1

        def rhs(u, flat, sign=sign):
            current = flat.reshape(2, 2, -1)
            t, jac = pulse.tau_us * np.sinh(u), pulse.tau_us * np.cosh(u)
            omega = 2 * np.pi * rabi_mhz * pulse.signed_envelope(t, sign)
            g, e = current
            return (-0.5j * jac * np.array([
                detuning * g + omega * e,
                omega * g - detuning * e,
            ])).ravel()

        solution = solve_ivp(rhs, (start, stop), state.ravel(), method="DOP853",
                             rtol=rtol, atol=rtol * 0.01)
        if not solution.success:
            raise RuntimeError(solution.message)
        state = solution.y[:, -1].reshape(2, 2, -1)
    return state


def population(pulse, rabi_mhz, detuning_mhz, **kwargs):
    return abs(propagator(pulse, rabi_mhz, detuning_mhz, **kwargs)[1, 0])**2


def rotation_integrals(pulse, rabi_mhz):
    """Quadrature of theta,S,C and the two first-order dissipative weights."""
    state = np.zeros(5)
    bounds = pulse.boundaries()
    for start, stop in zip(bounds[:-1], bounds[1:], strict=True):
        sign = 1 if start < 0 else -1

        def rhs(u, state, sign=sign):
            theta = state[0]
            t, jac = pulse.tau_us * np.sinh(u), pulse.tau_us * np.cosh(u)
            omega = 2 * np.pi * rabi_mhz * pulse.signed_envelope(t, sign)
            return jac * np.array([omega, np.sin(theta), np.cos(theta),
                                   np.sin(theta / 2)**4, np.sin(theta)**2 / 2])

        solution = solve_ivp(rhs, (start, stop), state, method="DOP853",
                             rtol=2e-11, atol=2e-12)
        if not solution.success:
            raise RuntimeError(solution.message)
        state = solution.y[:, -1]
    return dict(zip(["area", "S_us", "C_us", "relaxation_us", "dephasing_us"],
                    state, strict=True))


def bloch_population(pulse, rabi_mhz, detuning_mhz, gamma1, gammaphi, rtol=2e-10):
    """Population with Markovian rates in inverse microseconds."""
    detuning = 2 * np.pi * np.atleast_1d(detuning_mhz)
    state = np.zeros((3, detuning.size))
    state[2] = 1
    gamma2 = gamma1 / 2 + gammaphi
    bounds = pulse.boundaries()
    for start, stop in zip(bounds[:-1], bounds[1:], strict=True):
        sign = 1 if start < 0 else -1

        def rhs(u, flat, sign=sign):
            xx, yy, zz = flat.reshape(3, -1)
            t, jac = pulse.tau_us * np.sinh(u), pulse.tau_us * np.cosh(u)
            omega = 2 * np.pi * rabi_mhz * pulse.signed_envelope(t, sign)
            return (jac * np.array([
                -detuning * yy - gamma2 * xx,
                detuning * xx - omega * zz - gamma2 * yy,
                omega * yy + gamma1 * (1 - zz),
            ])).ravel()

        solution = solve_ivp(rhs, (start, stop), state.ravel(), method="DOP853",
                             rtol=rtol, atol=rtol * 0.01)
        if not solution.success:
            raise RuntimeError(solution.message)
        state = solution.y[:, -1].reshape(3, -1)
    return (1 - state[2]) / 2
