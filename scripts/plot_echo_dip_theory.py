"""Generate formula-versus-dynamics figures for paper/notes/echo_dip_theory.tex."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import find_peaks

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from echospec.simulation.echo_theory import (  # noqa: E402
    EchoTheoryPulse,
    adiabatic_probability,
    bloch_population,
    population,
    propagator,
    root_curvature_times,
    rotation_integrals,
)

FIGURES = ROOT / "figures/paper/echo_dip_theory"
DATA = ROOT / "outputs/echo_dip_theory"
ARRAYS = {}
SUMMARY = {"model": "Ideal two-level RWA; no DRAG, Stark shift, or leakage"}
COLORS = ["#176b9b", "#df7b24", "#23846f", "#9b477b"]


def save(fig, name):
    fig.savefig(FIGURES / f"{name}.png", dpi=180, bbox_inches="tight")
    fig.savefig(FIGURES / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {name}", flush=True)


def crossing(x, y, height):
    indices = np.flatnonzero((y[:-1] < height) & (y[1:] >= height))
    if not indices.size:
        return np.nan
    i = indices[0]
    return x[i] + (height - y[i]) / (y[i + 1] - y[i]) * (x[i + 1] - x[i])


def widths(x, y):
    """Symmetric dip: unit-shoulder crossing and first-maximum half depth.

    x must be nonnegative and y(0) must be the intended central minimum.
    The first shoulder is the first local maximum with prominence > 0.001.
    Missing shoulders/crossings remain NaN; never use a window endpoint.
    """
    peaks, _ = find_peaks(y, prominence=0.001)
    unit = 2 * crossing(x, y, (1 + y[0]) / 2)
    if not peaks.size:
        return unit, np.nan, np.nan
    peak = peaks[0]
    coef = np.polyfit(x[peak - 1:peak + 2] - x[peak], y[peak - 1:peak + 2], 2)
    shoulder = coef[2] - coef[1]**2 / (4 * coef[0])
    half = y[0] + (shoulder - y[0]) / 2
    width = 2 * crossing(x[:peak + 1], y[:peak + 1], half)
    return unit, width, shoulder


def mirror(x, y):
    return np.r_[-x[:0:-1], x], np.r_[y[:0:-1], y]


def mechanism():
    pulse = EchoTheoryPulse()
    a = 3.0
    rabi = a / (2 * np.pi * pulse.tau_us)
    u = np.linspace(-pulse.u_limit, pulse.u_limit, 2001)
    t = pulse.tau_us * np.sinh(u)
    omega = pulse.signed_envelope(t)
    theta = a * (pulse.u_limit - abs(u))
    fig, axes = plt.subplots(2, 2, figsize=(9.2, 6.8), layout="constrained")
    ax = axes[0, 0]
    ax.plot(t, omega, color=COLORS[0])
    ax.set_xscale("symlog", linthresh=pulse.tau_us)
    ax.set_xticks([-10, -1, -0.1, 0, 0.1, 1, 10],
                  ["-10", "-1", "-0.1", "0", "0.1", "1", "10"])
    ax.set(xlabel=r"Time $t$ ($\mu$s; symmetric-log scale)",
           ylabel=r"$\Omega_E(t)/\Omega_0$", title="(a) Opposite signed areas")
    ax.axhline(0, color="0.6", lw=0.7)
    ax.text(0.04, 0.08, r"$\int\Omega_E(t)dt=0\ \Rightarrow\ P_e(0)=0$",
            transform=ax.transAxes, fontsize=10)
    axes[0, 1].plot(t, theta, color=COLORS[2])
    axes[0, 1].set(xlabel=r"Time $t$ ($\mu$s)", ylabel=r"$\theta(t)$ (rad)",
                   title=r"(b) $\theta(t)=\int_{-L/2}^t\Omega_E(t')dt'$ ")
    delta = np.linspace(0, 0.002, 81)
    local = population(pulse, rabi, delta)
    s, _ = root_curvature_times(pulse, rabi)
    quadratic = (2 * np.pi * delta * s / 2)**2
    axes[1, 0].plot(delta * 1000, local, color=COLORS[0], label="Schrodinger equation")
    axes[1, 0].plot(delta * 1000, quadratic, "--", color=COLORS[1],
                    label=r"$P_e=\Delta^2S^2/4$")
    axes[1, 0].set(xlabel="Detuning (kHz)", ylabel=r"$P_e$",
                   title="(c) Exact curvature, local approximation")
    axes[1, 0].legend(fontsize=8)
    amps = np.geomspace(0.05, 30, 200)
    s_all, _ = root_curvature_times(pulse, amps / (2 * np.pi * pulse.tau_us))
    samples = np.geomspace(0.06, 25, 11)
    numerical = np.array([rotation_integrals(pulse, aa / (2 * np.pi * pulse.tau_us))["S_us"]
                          for aa in samples])
    axes[1, 1].semilogx(amps, s_all / pulse.duration_us, color=COLORS[0],
                        label="Exact closed form")
    axes[1, 1].semilogx(amps, amps / (1 + amps**2), "--", color=COLORS[1],
                        label=r"$S/L\simeq a/(1+a^2)$")
    axes[1, 1].scatter(samples, numerical / pulse.duration_us, s=20,
                       facecolors="none", edgecolors=COLORS[2], label="Rotation integral")
    axes[1, 1].set(xlabel=r"$a=\Omega_0\tau$", ylabel=r"$S/L$",
                   title="(d) Drive dependence of dip curvature")
    axes[1, 1].legend(fontsize=8)
    for ax in axes.flat:
        ax.grid(alpha=0.18)
    fig.suptitle(r"Echo cancellation and curvature: $L=20\,\mu$s, $c=0.005$")
    ARRAYS.update(mechanism_t_us=t, mechanism_omega=omega, mechanism_theta=theta,
                  curvature_detuning_mhz=delta, curvature_population=local,
                  curvature_quadratic=quadratic, curvature_a=amps, curvature_S_us=s_all,
                  curvature_sample_a=samples, curvature_sample_S_us=numerical)
    SUMMARY["curvature_quadrature_max_abs_error_us"] = float(np.max(abs(
        numerical - root_curvature_times(pulse, samples / (2 * np.pi * pulse.tau_us))[0]
    )))
    save(fig, "01_mechanism_curvature")


def lineshapes():
    pulse = EchoTheoryPulse()
    x = np.linspace(0, 4, 401)
    fig, axes = plt.subplots(2, 2, figsize=(9.2, 6.4), sharex=True, sharey=True,
                             layout="constrained")
    identity_errors, refinement_errors = [], []
    for a, ax in zip([0.3, 1, 3, 10], axes.flat, strict=True):
        rabi = a / (2 * np.pi * pulse.tau_us)
        det = x * rabi * pulse.cutoff
        full = propagator(pulse, rabi, det)
        exact = abs(full[1, 0])**2
        adiabatic = adiabatic_probability(pulse, rabi, det)
        lorentzian = x**2 / (1 + x**2)
        half = propagator(pulse, rabi, det[::40], half_only=True)
        identity = 4 * np.real(half[0, 0] * half[0, 1])**2
        identity_errors.append(float(np.max(abs(identity - exact[::40]))))
        refined = population(pulse, rabi, det[::40], rtol=2e-12)
        refinement_errors.append(float(np.max(abs(refined - exact[::40]))))
        ax.plot(*mirror(x, exact), color=COLORS[0], label="Full numerical dynamics")
        ax.plot(*mirror(x, adiabatic), "--", color=COLORS[1], label="Adiabatic-half formula")
        ax.plot(*mirror(x, lorentzian), ":", color=COLORS[2], lw=2,
                label=r"$P_e=x^2/(1+x^2)$")
        ax.axhline(0.5, color="0.7", lw=0.7)
        ax.axvline(1, color="0.75", lw=0.7)
        ax.axvline(-1, color="0.75", lw=0.7)
        ax.set_title(rf"$a={a:g}$; $f_{{\rm Rabi}}={rabi:.2f}$ MHz")
        ax.set_ylim(-0.02, 1.03)
        ax.grid(alpha=0.18)
        ARRAYS[f"spectrum_a{a:g}_numerical"] = exact
        ARRAYS[f"spectrum_a{a:g}_adiabatic"] = adiabatic
    for ax in axes[-1]:
        ax.set_xlabel(r"$x=\Delta/(\Omega_0c)$")
    for ax in axes[:, 0]:
        ax.set_ylabel(r"$P_e$")
    axes[0, 0].legend(fontsize=8)
    fig.suptitle("The Lorentzian dip emerges when the pulse halves are adiabatic")
    ARRAYS["spectrum_x"] = x
    SUMMARY["exact_half_identity_max_population_error"] = max(identity_errors)
    SUMMARY["solver_refinement_max_population_error"] = max(refinement_errors)
    assert max(identity_errors) < 1e-7
    assert max(refinement_errors) < 1e-7
    save(fig, "02_lineshapes")


def linewidths():
    pulse = EchoTheoryPulse()
    aa = np.geomspace(0.25, 25, 27)
    rows = []
    refinement_errors = []
    for a in aa:
        rabi = a / (2 * np.pi * pulse.tau_us)
        x = np.linspace(0, max(10, 8 / a), 1201)
        y = population(pulse, rabi, x * rabi * pulse.cutoff)
        unit, shoulder_width, shoulder = widths(x, y)
        predicted = 2 * rabi * pulse.cutoff
        rows.append([a, rabi, predicted, unit / 2 * predicted,
                     shoulder_width / 2 * predicted, shoulder])
        if a in aa[[0, 13, -1]]:
            xf = np.linspace(0, x[-1], 2401)
            yf = population(pulse, rabi, xf * rabi * pulse.cutoff, rtol=2e-12)
            finer = widths(xf, yf)
            errors = [float(abs(finer[i] / v - 1))
                      for i, v in enumerate([unit, shoulder_width]) if np.isfinite(v)]
            refinement_errors.extend(errors)
    assert max(refinement_errors) < 0.003
    SUMMARY["root_width_refinement_max_relative_error"] = max(refinement_errors)
    rows = np.asarray(rows)
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.9), layout="constrained")
    axes[0].loglog(rows[:, 1], 1000 * rows[:, 2], "--", color=COLORS[1],
                   label=r"$\Gamma_f=2c f_{\rm Rabi}$")
    axes[0].loglog(rows[:, 1], 1000 * rows[:, 3], "o-", ms=3, color=COLORS[0],
                   label=r"Numerical $P_e=0.5$ width")
    axes[0].loglog(rows[:, 1], 1000 * rows[:, 4], "s-", ms=3, color=COLORS[2],
                   label="First-shoulder half-depth width")
    axes[0].set(xlabel=r"$f_{\rm Rabi}$ (MHz)", ylabel="Full width (kHz)",
                 title="(a) Width depends on the shoulder definition")
    axes[0].legend(fontsize=8)
    axes[1].semilogx(aa, rows[:, 3] / rows[:, 2], "o-", ms=3, color=COLORS[0],
                     label=r"$P_e=0.5$ width / formula")
    axes[1].semilogx(aa, rows[:, 4] / rows[:, 2], "s-", ms=3, color=COLORS[2],
                     label="First-shoulder width / formula")
    axes[1].axhspan(0.95, 1.05, color="0.9", label="Within 5%")
    axes[1].axhline(1, color="0.4", lw=0.8)
    axes[1].set(xlabel=r"$a=\Omega_0\tau$", ylabel="Numerical / asymptotic width",
                 title="(b) Validity of the cutoff law", ylim=(0.65, 6.5), yscale="log")
    axes[1].legend(fontsize=8)
    for ax in axes:
        ax.grid(alpha=0.18)
    ARRAYS["root_width_rows"] = rows
    np.savetxt(DATA / "root_widths.csv", rows, delimiter=",",
               header="a,rabi_mhz,predicted_mhz,unit_shoulder_width_mhz,first_shoulder_width_mhz,first_shoulder_Pe",
               comments="")
    save(fig, "03_linewidth")


def shaping():
    settings = dict(duration_us=30, cutoff=0.001)
    pulses = [EchoTheoryPulse(**settings), EchoTheoryPulse(**settings, transition_us=0.016),
              EchoTheoryPulse(**settings, power=5, transition_us=0.016)]
    labels = ["Hard cutoff, abrupt echo", "Hard cutoff, 16 ns echo",
              "Polynomial p=5, 16 ns echo"]
    rabis = np.geomspace(10, 100, 13)
    delta = np.linspace(0, 0.8, 1601)
    records, spectra = [], []
    curvature_errors, grid_errors = [], []

    def window_width(x, y):
        # Equal shoulders at +/-0.4--0.5 MHz: the even-spectrum linear
        # background fit is exactly a constant. This is not a unit baseline.
        background = np.mean(y[(x >= 0.4) & (x <= 0.5)])
        return 2 * crossing(x, y, (background + y[0]) / 2), background

    for k, pulse in enumerate(pulses):
        curve = []
        for rabi in rabis:
            y = population(pulse, rabi, delta)
            width, shoulder = window_width(delta, y)
            curve.append(width)
            records.append([k, rabi, width, shoulder])
            if np.isclose(rabi, 100):
                spectra.append(y)
                refined_delta = np.linspace(0, 0.8, 3201)
                refined = population(pulse, rabi, refined_delta, rtol=2e-12)
                refined_width = window_width(refined_delta, refined)[0]
                grid_errors.append(float(abs(refined_width - width) / width))
                integral = rotation_integrals(pulse, rabi)
                tiny = 1e-5
                expected = (np.pi * tiny * integral["S_us"])**2
                actual = population(pulse, rabi, [tiny], rtol=2e-12)[0]
                curvature_errors.append(float(abs(actual / expected - 1)))
        ARRAYS[f"shaping_width_{k}"] = np.asarray(curve)
    assert np.all(np.isfinite(grid_errors)) and max(grid_errors) < 0.003
    assert np.all(np.isfinite(curvature_errors)) and max(curvature_errors) < 1e-4
    fig, axes = plt.subplots(2, 2, figsize=(9.2, 6.7), layout="constrained")
    s_over_h = np.geomspace(1e-3, 0.8, 300)
    t = 15 * (1 - s_over_h)
    exact_envelope = pulses[2].envelope(t)
    local_envelope = 0.001 * 1.001 * 2**5 * s_over_h**6
    axes[0, 0].loglog(s_over_h, exact_envelope, color=COLORS[2], label="Actual polynomial envelope")
    axes[0, 0].loglog(s_over_h, local_envelope, "--", color=COLORS[1],
                     label=r"$g\simeq c(1+c)2^5(s/h)^6$")
    axes[0, 0].set(xlabel=r"Distance from edge $s/h$", ylabel=r"$g(t)$",
                   title="(a) The sixth-order edge approximation")
    axes[0, 0].legend(fontsize=8)
    for k, (label, y) in enumerate(zip(labels, spectra, strict=True)):
        axes[0, 1].plot(*mirror(delta * 1000, y), color=COLORS[k], label=label)
        axes[1, 0].loglog(rabis, 1000 * ARRAYS[f"shaping_width_{k}"], "o-",
                          color=COLORS[k], ms=3, label=label)
        ARRAYS[f"shaping_spectrum_{k}"] = y
    axes[0, 1].set(xlabel="Detuning (kHz)", ylabel=r"$P_e$", xlim=(-400, 400),
                   title=r"(b) $f_{\rm Rabi}=100$ MHz; ideal two-level")
    axes[0, 1].legend(fontsize=7)
    poly_width = ARRAYS["shaping_width_2"]
    reference = poly_width[-1] * (rabis / rabis[-1])**(1 / 7)
    axes[1, 0].loglog(rabis, 1000 * reference, ":", color=COLORS[3], lw=2,
                      label=r"$f_{\rm Rabi}^{1/7}$, anchored at 100 MHz")
    axes[1, 0].set(xlabel=r"$f_{\rm Rabi}$ (MHz)", ylabel="Window-background FWHM (kHz)",
                   title="(c) Edge smoothing changes the width")
    axes[1, 0].legend(fontsize=7)
    a = 2 * np.pi * rabis * pulses[0].tau_us
    edge_length = (2 * np.pi * rabis * 15 * 0.001 * 1.001 * 2**5)**(-1 / 7)
    slope = np.gradient(np.log(poly_width), np.log(rabis))
    axes[1, 1].semilogx(rabis, slope, "o-", color=COLORS[2], ms=3,
                        label=r"Local slope $d\log\Gamma/d\log f_{\rm Rabi}$")
    axes[1, 1].axhline(1 / 7, color=COLORS[3], ls=":", label="Asymptotic exponent 1/7")
    axes[1, 1].semilogx(rabis, edge_length, "--", color="0.35", label=r"Edge scale $\ell/h$ (must be $\ll1$)")
    axes[1, 1].set(xlabel=r"$f_{\rm Rabi}$ (MHz)", ylabel="Dimensionless",
                   title="(d) Is the asymptotic regime reached?")
    axes[1, 1].legend(fontsize=7)
    for ax in axes.flat:
        ax.grid(alpha=0.18)
    fig.suptitle(r"Pulse modifications: $L=30\,\mu$s, $c=0.001$, no DRAG or dissipation")
    ARRAYS.update(shaping_rabi_mhz=rabis, shaping_delta_mhz=delta,
                  shaping_edge_s_over_h=s_over_h, shaping_edge_exact=exact_envelope,
                  shaping_edge_local=local_envelope, shaping_edge_length=edge_length,
                  shaping_slope=slope, shaping_width_rows=np.asarray(records))
    SUMMARY["shaping"] = {"duration_us": 30, "cutoff": 0.001,
                          "background_band_mhz": [0.4, 0.5],
                          "curvature_relative_errors": curvature_errors,
                          "width_grid_refinement_relative_errors": grid_errors,
                          "polynomial_local_slope_range": [float(min(slope)), float(max(slope))],
                          "edge_length_over_half_duration_range": [float(min(edge_length)), float(max(edge_length))],
                          "a_range": [float(min(a)), float(max(a))]}
    np.savetxt(DATA / "shaping_widths.csv", records, delimiter=",",
               header="pulse_index,rabi_mhz,window_background_width_mhz,background_Pe", comments="")
    save(fig, "04_shaping")


def dissipation():
    pulse = EchoTheoryPulse()
    rabi = 3 / (2 * np.pi * pulse.tau_us)
    gamma1 = 1 / 51.24
    gammaphi = 1 / 7.31 - gamma1 / 2
    integral = rotation_integrals(pulse, rabi)
    scales = np.geomspace(1e-4, 1, 25)
    predicted = scales * (gamma1 * integral["relaxation_us"] + gammaphi * integral["dephasing_us"])
    exact = np.array([bloch_population(pulse, rabi, [0], gamma1 * s, gammaphi * s)[0]
                      for s in scales])
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.9), layout="constrained")
    axes[0].loglog(scales, exact, "o-", color=COLORS[0], ms=3, label="Bloch equations")
    axes[0].loglog(scales, predicted, "--", color=COLORS[1], label="First-order jump formula")
    axes[0].set(xlabel=r"Rate multiplier $\lambda$", ylabel=r"On-resonance $P_e(0)$",
                 title="(a) Decoherence fills the resonant zero")
    axes[0].legend(fontsize=8)
    delta = np.linspace(0, 0.25, 501)
    for color, scale in zip(COLORS, [0, 0.1, 1], strict=False):
        y = bloch_population(pulse, rabi, delta, gamma1 * scale, gammaphi * scale)
        axes[1].plot(*mirror(delta * 1000, y), color=color, label=rf"$\lambda={scale:g}$")
        ARRAYS[f"dissipative_spectrum_{scale:g}"] = y
    axes[1].set(xlabel="Detuning (kHz)", ylabel=r"$P_e$",
                 title="(b) The shoulders and dip depth also change")
    axes[1].legend(fontsize=8)
    for ax in axes:
        ax.grid(alpha=0.18)
    fig.suptitle(r"At $\lambda=1$: $T_1=51.24\,\mu$s, $T_2=7.31\,\mu$s; $a=3$")
    ARRAYS.update(dissipation_scales=scales, dissipation_floor_exact=exact,
                  dissipation_floor_first_order=predicted, dissipation_delta_mhz=delta)
    SUMMARY["dissipation"] = {"T1_us_at_lambda_1": 51.24, "T2_us_at_lambda_1": 7.31,
                              "floor_exact_at_lambda_1": float(exact[-1]),
                              "floor_first_order_at_lambda_1": float(predicted[-1]),
                              "weakest_rate_relative_error": float(abs(exact[0] / predicted[0] - 1))}
    save(fig, "05_dissipation")


def main():
    FIGURES.mkdir(parents=True, exist_ok=True)
    DATA.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size": 10, "axes.titlesize": 11, "legend.fontsize": 8,
                         "axes.spines.top": False, "axes.spines.right": False})
    for make in [mechanism, lineshapes, linewidths, shaping, dissipation]:
        make()
    SUMMARY["root_pulse"] = {"duration_us": 20, "cutoff": 0.005,
                             "a_values_lineshapes": [0.3, 1, 3, 10],
                             "linewidth_a_range": [0.25, 25], "linewidth_a_count": 27}
    SUMMARY["solver"] = {"method": "DOP853 in u=asinh(t/tau)",
                         "relative_tolerance": 2e-10, "absolute_tolerance": 2e-12,
                         "refined_relative_tolerance": 2e-12,
                         "refined_absolute_tolerance": 2e-14}
    np.savez_compressed(FIGURES / "theory_arrays.npz", **ARRAYS)
    SUMMARY["width_definitions"] = {
        "unit_shoulder": "Twice nearest positive crossing at (1+Pe(0))/2; only a FWHM for unit shoulders",
        "first_shoulder": "Twice nearest positive half-depth crossing relative to first local peak; prominence >0.001",
        "shaping_window": "Half-depth crossings relative to the even-spectrum mean at abs(detuning)=0.4--0.5 MHz",
        "missing_values": "NaN when the requested shoulder or crossing is absent from the scan",
    }
    SUMMARY["sha256"] = {
        str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in [Path(__file__), ROOT / "echospec/simulation/echo_theory.py",
                  FIGURES / "theory_arrays.npz"]
    }
    (FIGURES / "provenance.json").write_text(json.dumps(SUMMARY, indent=2, allow_nan=False) + "\n")
    print(json.dumps(SUMMARY, indent=2))


if __name__ == "__main__":
    main()
