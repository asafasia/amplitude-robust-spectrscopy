"""Experimental single-reversal echo with independently shaped weak tails.

Units and full-waveform DRAG match PolynomialEchoPulse. The local-adiabatic
tail is inspired by integrating dOmega/dt proportional to
(Omega**2 + Delta_star**2)**(3/2), then regularized at the core and endpoints.
It is a candidate family, not an assertion of adiabaticity or optimality.
"""

from dataclasses import dataclass
from math import comb

import numpy as np

from .polynomial_echo import PolynomialEchoPulse


@dataclass(frozen=True)
class SelectiveEchoPulse(PolynomialEchoPulse):
    tail_scale_mhz: float = 0.08
    tail_cap_mhz: float = 1.0
    core_us: float = 0.015
    edge_power: float = 2.0
    reference_rabi_mhz: float = 50.0

    def __post_init__(self):
        super().__post_init__()
        if not (0 < self.tail_scale_mhz and 0 < self.tail_cap_mhz
                < self.reference_rabi_mhz and 0 < self.core_us
                < self.duration_us / 2 and self.edge_power >= 1):
            raise ValueError("Invalid selective echo shape parameters")

    @property
    def sigma_us(self):
        return self.core_us

    def envelope(self, time_us):
        t = np.asarray(time_us, dtype=float)
        h = self.duration_us / 2
        radius = np.hypot(t, self.core_us) - self.core_us
        span = np.hypot(h, self.core_us) - self.core_us
        z = np.clip(1 - radius / span, 0, 1)
        dz = -t / (np.hypot(t, self.core_us) * span)
        umax = 1 / np.hypot(1, self.tail_scale_mhz / self.tail_cap_mhz)
        u = umax * z**self.edge_power
        du = umax * self.edge_power * z**(self.edge_power - 1) * dz
        denom = np.sqrt(1 - u * u)
        tail = self.tail_scale_mhz * u / denom
        dtail = self.tail_scale_mhz * du / denom**3
        core = np.exp(-(t / self.core_us)**2)
        dcore = -2 * t / self.core_us**2 * core
        scale = self.reference_rabi_mhz
        g = (tail + (scale - self.tail_cap_mhz) * core) / scale
        dg = (dtail + (scale - self.tail_cap_mhz) * dcore) / scale
        inside = abs(t) < h
        return np.where(inside, g, 0), np.where(inside, dg, 0)


@dataclass(frozen=True)
class PhaseEchoPulse(PolynomialEchoPulse):
    """Even phase modulation preserves odd complex control before DRAG.

    In the ideal resonant two-level model, Hc(-t)=-Hc(t) gives U=identity
    for every global amplitude scale, even when control axes do not commute.
    Multilevel effects and derivative correction are tested by propagation.
    """

    phase_rad: float = 0.0
    phase_scale_us: float = 2.0

    def __post_init__(self):
        super().__post_init__()
        if self.phase_scale_us <= 0:
            raise ValueError("Phase scale must be positive")

    def drive(self, time_us):
        t = np.asarray(time_us, dtype=float)
        base = super().drive(t)
        phase = self.phase_rad * np.exp(-(t / self.phase_scale_us)**2)
        dphase = -2 * t / self.phase_scale_us**2 * phase
        # base=I+i*beta*I'/|alpha|; apply the derivative to I*exp(i*phase).
        correction = self.drag_beta * dphase / (2 * np.pi * abs(self.anharmonicity_mhz))
        return np.exp(1j * phase) * (base - correction * base.real)


def simulate_echo_split(pulse, amplitude_mhz, detuning_mhz, *, levels=4,
                        t1_us=51.24, t2_us=7.31, refinement=1):
    """Independent CPTP midpoint/Strang check with exact damping substeps.

    A finite Duffing ladder, truncated oscillator loss, and number dephasing
    match the native qutrit equations. Time refinement is still required.
    The amplitude scales pulse.drive directly (no internal normalization).
    """
    detuning = np.asarray(detuning_mhz, dtype=float)
    if levels < 2 or refinement < 1 or not 0 < t2_us <= 2 * t1_us:
        raise ValueError("Invalid level, refinement, or coherence setting")
    half = pulse.duration_us / 2
    core_end = min(.1, half)
    positive = list(np.linspace(0, core_end, int(np.ceil(core_end / .0005 * refinement)) + 1))
    while positive[-1] < half:
        current = positive[-1]
        positive.append(min(half, current + min(.02, .01 * current) / refinement))
    positive = np.unique(np.r_[positive, pulse.transition_us / 2])
    grid = np.r_[-positive[:0:-1], positive]
    number = np.arange(levels)
    rho = np.zeros((len(detuning), levels, levels), dtype=complex)
    rho[:, 0, 0] = 1
    inv_phi = 1 / t2_us - 1 / (2 * t1_us)
    diff = number[:, None] - number[None, :]
    energies = 2 * np.pi * (-detuning[:, None] * number +
                           pulse.anharmonicity_mhz / 2 * number * (number - 1))

    def damping(state, step):
        eta = np.exp(-step / t1_us)
        output = np.zeros_like(state)
        for loss in range(levels):
            size = levels - loss
            factors = np.array([np.sqrt(comb(m + loss, loss) *
                                        (1 - eta)**loss * eta**m) for m in range(size)])
            output[:, :size, :size] += (state[:, loss:, loss:] *
                                        factors[:, None] * factors[None, :])
        return output * np.exp(-step * inv_phi * diff**2)

    for start, stop in zip(grid[:-1], grid[1:], strict=True):
        step = stop - start
        hamiltonian = np.zeros_like(rho)
        hamiltonian[:, number, number] = energies
        coupling = np.pi * amplitude_mhz * pulse.drive((start + stop) / 2)
        for lower in range(levels - 1):
            hamiltonian[:, lower, lower + 1] = coupling * np.sqrt(lower + 1)
            hamiltonian[:, lower + 1, lower] = coupling.conjugate() * np.sqrt(lower + 1)
        eigenvalues, vectors = np.linalg.eigh(hamiltonian)
        unitary = (vectors * np.exp(-1j * step * eigenvalues)[:, None, :]) @ vectors.conj().transpose(0, 2, 1)
        rho = damping(rho, step / 2)
        rho = unitary @ rho @ unitary.conj().transpose(0, 2, 1)
        rho = damping(rho, step / 2)
    return rho.diagonal(axis1=1, axis2=2).real.T
