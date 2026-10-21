"""Verify the standalone echo-dip derivation with ideal two-level dynamics.

Run with .venv/bin/python scripts/check_echo_dip_theory.py.
Outputs are theoretical checks, not experimental or qutrit validation.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import quad, solve_ivp
from scipy.linalg import expm
from scipy.optimize import brentq


def root_response(a, cutoff, x, rtol=2e-10):
    """Return Pe for x = Delta/(Omega0*c), using u = asinh(t/tau)."""
    x = np.atleast_1d(x)
    limit = np.arccosh(1 / cutoff)
    state = np.zeros((2, x.size), dtype=complex)
    state[0] = 1

    for start, stop, sign in [(-limit, 0, 1), (0, limit, -1)]:
        def rhs(u, flat, sign=sign):
            g, e = flat.reshape(2, -1)
            diagonal = a * cutoff * x * np.cosh(u)
            return (-0.5j * np.array([
                diagonal * g + sign * a * e,
                sign * a * g - diagonal * e,
            ])).ravel()

        result = solve_ivp(
            rhs, (start, stop), state.ravel(), method="DOP853",
            rtol=rtol, atol=rtol * 0.01,
        )
        if not result.success:
            raise RuntimeError(result.message)
        state = result.y[:, -1].reshape(2, -1)
    np.testing.assert_allclose(np.sum(abs(state) ** 2, axis=0), 1, atol=2e-8)
    return abs(state[1]) ** 2


def adiabatic_response(a, cutoff, x):
    limit = np.arccosh(1 / cutoff)
    phase = np.array([
        quad(lambda u, xx=xx: a * np.sqrt(1 + (cutoff * xx * np.cosh(u)) ** 2),
             0, limit, epsabs=1e-10)[0]
        for xx in x
    ])
    return (
        x ** 2 / (1 + x ** 2)
        * (1 - cutoff * np.cos(phase)) ** 2
        / (1 + (cutoff * x) ** 2)
    )


def check_square_echo():
    """Check the half-propagator algebra against exact matrix exponentials."""
    sx = np.array([[0, 1], [1, 0]])
    sz = np.diag([1, -1])
    for omega, delta, length in [(2.1, 0.3, 4.2), (0.4, 1.7, 2.3)]:
        first = expm(-0.25j * length * (delta * sz + omega * sx))
        second = expm(-0.25j * length * (delta * sz - omega * sx))
        q = np.hypot(omega, delta)
        predicted = 4 * delta**2 * omega**2 / q**4 * np.sin(q * length / 4)**4
        np.testing.assert_allclose(abs((second @ first)[1, 0])**2,
                                   predicted, rtol=1e-13)


def check_dissipative_floor():
    """Verify the first-order floor and its quadratic remainder in the rates."""
    cutoff, a, length = 0.05, 1.3, 1.0
    tau = length * cutoff / (2 * np.sqrt(1 - cutoff**2))
    limit = np.arccosh(1 / cutoff)
    errors = []
    for scale in [1.0, 0.5]:
        gamma1, gammaphi = 1e-3 * scale, 2e-3 * scale
        gamma2 = gamma1 / 2 + gammaphi
        state = np.array([0.0, 0.0, 1.0])
        for start, stop, sign in [(-limit, 0, 1), (0, limit, -1)]:
            def rhs(u, state, sign=sign, gamma1=gamma1, gamma2=gamma2):
                xx, yy, zz = state
                jac = tau * np.cosh(u)
                return [-jac * gamma2 * xx,
                        -sign * a * zz - jac * gamma2 * yy,
                        sign * a * yy + jac * gamma1 * (1 - zz)]

            result = solve_ivp(rhs, (start, stop), state, method="DOP853",
                               rtol=1e-12, atol=1e-14)
            assert result.success
            state = result.y[:, -1]
        predicted = 2 * tau * quad(
            lambda u, gamma1=gamma1, gammaphi=gammaphi: np.cosh(u) * (
                gamma1 * np.sin(a * (limit - u) / 2)**4
                + gammaphi / 2 * np.sin(a * (limit - u))**2
            ), 0, limit, epsabs=1e-13,
        )[0]
        actual = (1 - state[2]) / 2
        np.testing.assert_allclose(actual, predicted, rtol=0.002)
        errors.append(abs(actual - predicted))
    ratio = errors[0] / errors[1]
    assert 3.9 < ratio < 4.1
    return {"rate_halving_remainder_ratio": ratio,
            "absolute_remainders": errors}


def main():
    output = Path(__file__).resolve().parents[1] / "outputs/echo_dip_theory"
    output.mkdir(parents=True, exist_ok=True)
    check_square_echo()
    floor_check = check_dissipative_floor()
    cutoff = 0.005
    duration_us = 20.0
    tau_us = duration_us * cutoff / (2 * np.sqrt(1 - cutoff ** 2))
    limit = np.arccosh(1 / cutoff)
    x = np.linspace(0, 4, 401)
    arrays = {"x_detuning_over_edge_rabi": x}
    rows = []
    fig, axes = plt.subplots(2, 2, figsize=(9, 6), sharex=True, sharey=True)
    for a, ax in zip([0.3, 1.0, 3.0, 10.0], axes.flat, strict=True):
        # S/tau is dimensionless and has an elementary closed form.
        s = 2 * a / (1 + a ** 2) * (1 / cutoff - np.cos(a * limit))
        integral = 2 * quad(
            lambda u, a=a: np.cosh(u) * np.sin(a * (limit - u)),
            0, limit, epsabs=1e-10,
        )[0]
        np.testing.assert_allclose(s, integral, rtol=2e-12, atol=1e-10)
        tiny_x = 1e-4
        local_prediction = (a * cutoff * tiny_x * s / 2) ** 2
        local_exact = root_response(a, cutoff, [tiny_x])[0]
        np.testing.assert_allclose(local_exact, local_prediction, rtol=2e-6)

        exact = root_response(a, cutoff, x)
        refined = root_response(a, cutoff, x[::20], rtol=2e-12)
        convergence_error = float(np.max(abs(exact[::20] - refined)))
        assert convergence_error < 2e-8
        np.testing.assert_allclose(exact[0], 0, atol=1e-18)
        adiabatic = adiabatic_response(a, cutoff, x)
        lorentzian = x ** 2 / (1 + x ** 2)
        crossings = np.flatnonzero((exact[:-1] < 0.5) & (exact[1:] >= 0.5))
        crossing = None
        if crossings.size:
            idx = crossings[0]
            crossing = brentq(
                lambda xx, a=a: root_response(a, cutoff, [xx])[0] - 0.5,
                x[idx], x[idx + 1], xtol=1e-10,
            )
        predicted_width_mhz = a * cutoff / (np.pi * tau_us)
        rows.append({
            "a_Omega0_tau": a,
            "peak_rabi_mhz": a / (2 * np.pi * tau_us),
            "S_us": s * tau_us,
            "curvature_relative_error": float(local_exact / local_prediction - 1),
            "numerical_refinement_max_population_error": convergence_error,
            "cutoff_lorentzian_width_khz": 1000 * predicted_width_mhz,
            "exact_Pe_half_crossing_width_khz": (
                None if crossing is None else 1000 * crossing * predicted_width_mhz
            ),
            "half_crossing_over_asymptotic": crossing,
        })
        arrays[f"Pe_exact_a{a:g}"] = exact
        arrays[f"Pe_adiabatic_a{a:g}"] = adiabatic
        both_x = np.r_[-x[:0:-1], x]
        ax.plot(both_x, np.r_[exact[:0:-1], exact],
                label="Two-level numerical", lw=2)
        ax.plot(both_x, np.r_[adiabatic[:0:-1], adiabatic], "--",
                label="Adiabatic halves", lw=1.4)
        ax.plot(both_x, np.r_[lorentzian[:0:-1], lorentzian], ":",
                label="Cutoff Lorentzian", lw=1.6)
        ax.axhline(0.5, color="0.7", lw=0.7)
        ax.set_title(rf"$a=\Omega_0\tau={a:g}$")
        ax.set_ylim(-0.02, 1.04)
        ax.grid(alpha=0.2)
    for ax in axes[-1]:
        ax.set_xlabel(r"$\Delta/(\Omega_0 c)$")
    for ax in axes[:, 0]:
        ax.set_ylabel(r"$P_e$")
    axes[0, 0].legend(fontsize=8)
    fig.suptitle("Ideal root-Lorentzian echo: c = 0.005; no decoherence or DRAG")
    fig.tight_layout()
    fig.savefig(output / "comparison.png", dpi=180)
    plt.close(fig)
    np.savez_compressed(output / "comparison.npz", **arrays)
    report = {
        "model": "Unitary two-level RWA; abrupt echo; ground-state preparation",
        "duration_us": duration_us,
        "cutoff": cutoff,
        "tau_us": tau_us,
        "weak_dissipation_floor_check": floor_check,
        "width_definition": (
            "Twice the first positive Pe=0.5 crossing, converted to cyclic frequency. "
            "This approximates a dip FWHM only when the local shoulder is near 1. "
            "Null means no crossing within 0 <= Delta/(Omega0*c) <= 4."
        ),
        "rows": rows,
    }
    (output / "checks.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
