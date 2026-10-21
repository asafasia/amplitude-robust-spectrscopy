"""Broad uncorrected spectroscopy and f01 Stark comparison; MHz and us units."""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from dataclasses import asdict, replace
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/ars-matplotlib-cache")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from scipy.ndimage import gaussian_filter1d  # noqa: E402

from echospec.simulation.config import SimulationConfig  # noqa: E402
from echospec.simulation.piecewise import (  # noqa: E402
    dressed_f01_center,
    simulate_pulse_pair,
)


def populations(states):
    return np.diagonal(states, axis1=-2, axis2=-1).real


def diagnostics(states):
    return {
        "trace_error": float(np.max(abs(np.trace(states, axis1=-2, axis2=-1) - 1))),
        "hermiticity_error": float(np.max(abs(states - states.swapaxes(-1, -2).conj()))),
        "minimum_eigenvalue": float(np.linalg.eigvalsh(states).min()),
    }


def peak_centers(x, values, sigma_mhz):
    """Operational P1 envelope maximum, with explicit frequency smoothing."""
    smooth = gaussian_filter1d(values, sigma_mhz / (x[1] - x[0]), axis=1)
    centers = np.full(values.shape[0], np.nan)
    for index in range(1, values.shape[0]):
        i = int(np.argmax(smooth[index]))
        if i in (0, len(x) - 1):
            continue
        left, center, right = smooth[index, i - 1:i + 2]
        offset = 0.5 * (left - right) / (left - 2 * center + right)
        centers[index] = x[i] + offset * (x[1] - x[0])
    return centers


def calculate(config, output, segments):
    amplitudes = np.asarray(config.amplitude_mhz)
    broad = np.asarray(config.detuning_mhz)
    zoom = np.linspace(-20, 75, 951)
    # The denser f01 window is simulated, not interpolated from the broad map.
    joint = np.unique(np.round(np.concatenate((broad, zoom)), 8))
    data = {"rabi_mhz": amplitudes, "detuning_mhz": broad, "zoom_detuning_mhz": zoom}
    checks = {}
    for name in ("ordinary", "echo"):
        data[f"{name}_population"] = np.empty((len(amplitudes), len(broad), config.levels))
        data[f"{name}_zoom_population"] = np.empty((len(amplitudes), len(zoom), config.levels))
        checks[name] = {"trace_error": 0.0, "hermiticity_error": 0.0, "minimum_eigenvalue": 0.0}
    start = time.perf_counter()
    for index, amplitude in enumerate(amplitudes):
        results = simulate_pulse_pair(
            replace(config, amplitude_mhz=(float(amplitude),), detuning_mhz=tuple(joint)),
            segments_per_half=segments,
        )
        for name, states in results.items():
            p = populations(states)[0]
            data[f"{name}_population"][index] = p[np.searchsorted(joint, np.round(broad, 8))]
            data[f"{name}_zoom_population"][index] = p[np.searchsorted(joint, np.round(zoom, 8))]
            diag = diagnostics(states)
            for key, value in diag.items():
                op = min if key == "minimum_eigenvalue" else max
                checks[name][key] = op(checks[name][key], value)
        print(f"Rabi {amplitude:5.1f} MHz: {index + 1}/{len(amplitudes)}, "
              f"{time.perf_counter() - start:.1f} s", flush=True)
    for name, diagnostic in checks.items():
        if (diagnostic["trace_error"] > 1e-7
                or diagnostic["hermiticity_error"] > 1e-7
                or diagnostic["minimum_eigenvalue"] < -1e-7):
            raise RuntimeError(f"Unphysical {name} states: {diagnostic}")
    np.savez_compressed(output / "spectroscopy.npz", **data)
    metadata = {
        "config": asdict(config), "segments_per_half": segments,
        "elapsed_seconds": time.perf_counter() - start,
        "diagnostics": checks, "detuning_convention": "drive_frequency_minus_bare_f01",
        "model": "4-level Duffing Lindblad, RWA, initial ground, no DRAG or compensation",
        "broad_step_mhz": float(broad[1] - broad[0]), "zoom_step_mhz": float(zoom[1] - zoom[0]),
        "note": "Final population samples; the broad grid does not resolve every coherent fringe.",
    }
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    return data, metadata


def validate(config, segments):
    probe = replace(config, amplitude_mhz=(10.0, 50.0, 100.0),
                    detuning_mhz=(-125.0, -108.55, -50.0, 0.0, 5.0, 17.0, 23.0, 50.0, 125.0))
    coarse = simulate_pulse_pair(probe, segments_per_half=segments)
    fine = simulate_pulse_pair(probe, segments_per_half=4 * segments)
    extra = simulate_pulse_pair(replace(probe, levels=5), segments_per_half=4 * segments)
    checks = {
        "probe_rabi_mhz": probe.amplitude_mhz,
        "probe_detuning_mhz": probe.detuning_mhz,
        "max_population_change_quadrupled_segments": max(
            float(np.max(abs(populations(coarse[k]) - populations(fine[k])))) for k in fine
        ),
        "max_P1_change_four_to_five_levels_on_probes": max(
            float(np.max(abs(populations(fine[k])[..., 1] - populations(extra[k])[..., 1]))) for k in fine
        ),
        "f01_center_change_four_to_five_levels_at_100mhz": abs(
            dressed_f01_center(100, config.anharmonicity_mhz, levels=4)
            - dressed_f01_center(100, config.anharmonicity_mhz, levels=5)
        ),
    }
    if checks["max_population_change_quadrupled_segments"] > 1e-4:
        raise RuntimeError(f"Envelope convergence insufficient: {checks}")
    return checks


def refine_centers(config, output, data, segments):
    """Resolve the P1 maximum on a 20 kHz grid before frequency smoothing."""
    coarse_x = data["zoom_detuning_mhz"]
    names = ("ordinary", "echo")
    sigmas = (0.25, 0.5, 1.0)
    initial = {
        f"{name}_peak_sigma_{sigma:g}": peak_centers(
            coarse_x, data[f"{name}_zoom_population"][..., 1], sigma
        ) for name in names for sigma in sigmas
    }
    refined = {key: np.full(len(config.amplitude_mhz), np.nan) for key in initial}
    saved = {"rabi_mhz": np.asarray(config.amplitude_mhz)}
    for i, amp in enumerate(config.amplitude_mhz[1:], 1):
        guesses = np.array([values[i] for values in initial.values()])
        lower = max(-20, np.nanmin(guesses) - 4)
        upper = min(75, np.nanmax(guesses) + 4)
        x = np.arange(np.floor(lower / 0.02), np.ceil(upper / 0.02) + 1) * 0.02
        results = simulate_pulse_pair(
            replace(config, amplitude_mhz=(float(amp),), detuning_mhz=tuple(x)),
            segments_per_half=segments,
        )
        saved[f"detuning_{i}"] = x
        for name, rho in results.items():
            values = populations(rho)[0]
            saved[f"{name}_population_{i}"] = values
            for sigma in sigmas:
                # peak_centers reserves its first row for zero drive.
                center = peak_centers(x, np.stack((np.zeros(len(x)), values[:, 1])), sigma)[1]
                if not np.isfinite(center) or min(center - x[0], x[-1] - center) < 2:
                    raise RuntimeError(f"Refinement window too narrow at {amp} MHz: {center}")
                refined[f"{name}_peak_sigma_{sigma:g}"][i] = center
        print(f"Refined P1 peaks: {amp:g} MHz ({i}/50)", flush=True)
    np.savez_compressed(output / "refined_peak_spectra.npz", **saved)
    np.savez_compressed(output / "refined_peak_centers.npz", **refined)
    return refined


def render(config, output, data, metadata):
    a, x, z = (data[k] for k in ("rabi_mhz", "detuning_mhz", "zoom_detuning_mhz"))
    edge = np.sqrt(config.cutoff**(-2) - 1)
    mean_square = np.arctan(edge) / edge
    theory = -a**2 * mean_square / (2 * config.anharmonicity_mhz)
    nodes, weights = np.polynomial.legendre.leggauss(24)
    scales = (1 + (nodes * edge)**2)**(-0.5)
    dressed = np.array([
        sum(w * dressed_f01_center(float(amp * s), config.anharmonicity_mhz,
                                  levels=config.levels) for s, w in zip(scales, weights, strict=True)) / 2
        for amp in a
    ])
    centers = {}
    for name in ("ordinary", "echo"):
        values = data[f"{name}_zoom_population"][..., 1]
        for sigma in (0.25, 0.5, 1.0):
            centers[f"{name}_peak_sigma_{sigma:g}"] = peak_centers(z, values, sigma)
    if (output / "refined_peak_centers.npz").exists():
        centers = dict(np.load(output / "refined_peak_centers.npz"))
    columns = {"rabi_mhz": a, "dressed_f01_shift_mhz": dressed,
               "quadratic_stark_shift_mhz": theory, **centers}
    with (output / "resonance_vs_amplitude.csv").open("w") as file:
        writer = csv.writer(file)
        writer.writerow(columns)
        writer.writerows(zip(*columns.values(), strict=True))
    np.savez_compressed(output / "resonance_analysis.npz", **columns)

    plt.rcParams.update({"font.size": 10, "axes.spines.top": False,
                         "axes.spines.right": False, "savefig.dpi": 190})
    fig, axs = plt.subplots(2, 2, figsize=(13.5, 9.2), layout="constrained")
    for ax, name in zip(axs[0], ("ordinary", "echo"), strict=True):
        mesh = ax.pcolormesh(x, a, data[f"{name}_population"][..., 1],
                             shading="auto", cmap="viridis", vmin=0, vmax=0.65, rasterized=True)
        ax.plot(dressed, a, color="white", lw=1.5, label="Dressed f01")
        ax.plot(theory, a, "--", color="#ffad67", lw=1.4, label="Quadratic theory")
        ax.axvline(0, color="white", ls=":", lw=0.9)
        ax.axvline(config.anharmonicity_mhz / 2, color="white", ls=":", lw=0.9)
        ax.set(xlim=(-125, 125), ylim=(0, 100), xlabel=r"Drive detuning $f_d-f_{01}$ (MHz)",
               ylabel=r"Peak Rabi frequency $\Omega_0/2\pi$ (MHz)",
               title=f"{'Ordinary pulse' if name == 'ordinary' else 'Echo: midpoint phase flip'} — final $P_1$")
        ax.legend(loc="upper left", fontsize=8, facecolor="#243344", labelcolor="white")
        fig.colorbar(mesh, ax=ax, label=r"$P_1$", fraction=0.045)
    ax = axs[1, 0]
    ax.plot(a, theory, "--", color="#cf652b", lw=2, label="Quadratic AC Stark estimate")
    ax.plot(a, dressed, color="#222222", lw=2.2, label="Dressed f01 (4 levels)")
    for name, color in (("ordinary", "#2076b4"), ("echo", "#14977c")):
        y = centers[f"{name}_peak_sigma_0.5"]
        ax.plot(a, y, color=color, lw=1.4, label=f"{name.capitalize()}: P1 envelope peak")
        bounds = np.stack([centers[f"{name}_peak_sigma_{s:g}"] for s in (0.25, 0.5, 1.0)])
        ax.fill_between(a, np.min(bounds, axis=0), np.max(bounds, axis=0), color=color, alpha=0.15)
    ax.set(xlabel="Peak Rabi frequency (MHz)", ylabel="Detuning of feature (MHz)",
           title="Resonance definition matters at strong drive", xlim=(0, 100))
    ax.grid(alpha=0.18)
    ax.legend(fontsize=8, loc="upper left")
    ax = axs[1, 1]
    for amp, color in ((20, "#627bbb"), (50, "#18a18d"), (100, "#d98232")):
        i = int(np.argmin(abs(a - amp)))
        ax.plot(z, data["echo_zoom_population"][i, :, 1], color=color, lw=0.9,
                label=f"Echo, {a[i]:g} MHz")
        ax.axvline(dressed[i], color=color, ls="--", lw=1)
    ax.set(xlabel=r"Drive detuning $f_d-f_{01}$ (MHz)", ylabel=r"Final $P_1$",
           title="Raw response; dashed lines = dressed f01", xlim=(-20, 75))
    ax.grid(alpha=0.18)
    ax.legend(fontsize=9)
    fig.suptitle(f"Cutoff {config.cutoff:g} · {config.duration_us:g} µs · 250 MHz span · Rabi 0–100 MHz\n"
                 f"α/2π = {config.anharmonicity_mhz:.3f} MHz; T1 = {config.t1_us:.2f} µs; "
                 f"T2 = {config.t2_us:.2f} µs; simulation in the rotating-wave approximation", fontsize=13)
    for suffix in ("png", "pdf"):
        fig.savefig(output / f"broad_spectroscopy.{suffix}")
    plt.close(fig)

    fig, axs = plt.subplots(2, 3, figsize=(14, 7), layout="constrained", sharex=True, sharey=True)
    for row, name in enumerate(("ordinary", "echo")):
        p = data[f"{name}_population"]
        for col, (values, label) in enumerate(((p[..., 0], "P0"), (p[..., 1], "P1"),
                                              (p[..., 2:].sum(axis=-1), "P2 + P3"))):
            ax = axs[row, col]
            im = ax.pcolormesh(x, a, values, shading="auto", cmap="viridis", vmin=0, vmax=1,
                              rasterized=True)
            ax.plot(dressed, a, "w-", lw=1)
            ax.axvline(config.anharmonicity_mhz / 2, color="white", ls=":", lw=1)
            ax.set(title=f"{name.capitalize()} — {label}", xlim=(-125, 125), ylim=(0, 100))
            if row == 1:
                ax.set_xlabel("Drive detuning (MHz)")
            if col == 0:
                ax.set_ylabel("Peak Rabi frequency (MHz)")
    fig.colorbar(im, ax=axs, label="Final population", fraction=0.02)
    fig.suptitle(f"{config.duration_us:g} µs, cutoff {config.cutoff:g}: population and leakage\n"
                 "White curve: dressed f01; dotted line: bare two-photon f02/2")
    fig.savefig(output / "state_populations.png")
    plt.close(fig)

    validation = metadata["validation"]
    rows = []
    for amp in (20, 50, 80, 100):
        i = int(np.argmin(abs(a - amp)))
        rows.append(f"| {a[i]:g} | {dressed[i]:.4f} | {theory[i]:.4f} | "
                    f"{centers['ordinary_peak_sigma_0.5'][i]:.4f} | {centers['echo_peak_sigma_0.5'][i]:.4f} |")
    report = f"""# {config.duration_us:g} µs near-square spectroscopy: cutoff {config.cutoff:g}

![Broad response and resonance comparison](broad_spectroscopy.png)

## Model and sweep

- Duration: {config.duration_us:g} µs; root-Lorentzian order 0.5; endpoint/peak = {config.cutoff:g}.
- Both ordinary and midpoint-pi-flipped echo pulses; no DRAG, Stark compensation, or endpoint subtraction.
- 51 peak Rabi amplitudes, 0–100 MHz; 501 broad detunings, −125 to +125 MHz.
- Additional 951 detunings from −20 to +75 MHz at 0.1 MHz spacing resolve the f01 region.
- Four-level Duffing transmon, Lindblad relaxation and number-operator pure dephasing; initial ground state.
- Current SimulationConfig values: alpha/2pi = {config.anharmonicity_mhz:.9f} MHz,
  T1 = {config.t1_us:.9f} µs, T2 = {config.t2_us:.9f} µs, Tphi = {config.t_phi_us:.9f} µs.
- f01 is the zero of the frequency axis. Add these shifts to your device's bare f01 to obtain an absolute frequency.
- Rotating-wave model: no Bloch–Siegert shift, resonator/readout response, hardware filtering, or noise realizations.

## What is being called resonance?

The solid dark curve is the drive detuning that minimizes the upper-two dressed-state
gap of H/h = −delta n + (alpha/2pi)n(n−1)/2 + A f(t)(a+a†)/2,
time averaged over the envelope. This follows the f01 avoided crossing. It is
independent of the sign flip and is not the largest point in a population spectrum.

The blue/green curves are operational maxima of final P1 in the f01 window,
after local refinement on a 0.02 MHz grid and Gaussian smoothing with sigma = 0.5 MHz.
Shading is the range from sigma
0.25 to 1 MHz, a smoothing-sensitivity range, not a statistical confidence interval.
These maxima include saturation, leakage, dissipation and finite-pulse effects;
they must not be identified automatically with the Hamiltonian AC Stark shift.
There is no measurable resonance at zero drive; operational centers there are NaN.
The broad overview samples do not resolve every finite-pulse fringe.

## Analytical comparison

With A = Omega0/2pi in MHz and alpha_f = alpha/2pi,

    delta_f_Stark ≈ −A² <f²> / (2 alpha_f)
    <f²> = atan(sqrt(c^(-2)−1)) / sqrt(c^(-2)−1) = {mean_square:.9f}

The sign is positive because alpha_f is negative. Flipping the drive phase
does not change f². At 100 MHz, A/|alpha_f| = {100/abs(config.anharmonicity_mhz):.3f},
so the weak-drive quadratic expression is no longer quantitatively accurate.
The comparison concerns a single near-resonant drive, not a separate off-resonant Stark tone.
The local derivation is in paper/appendices/K_ac_stark_shift.tex. Background on
multilevel dressing: [Multi-level quantum noise spectroscopy](https://pmc.ncbi.nlm.nih.gov/articles/PMC7878521/).

| Rabi (MHz) | Dressed f01 shift (MHz) | Quadratic (MHz) | Ordinary P1 peak (MHz) | Echo P1 peak (MHz) |
|---:|---:|---:|---:|---:|
{chr(10).join(rows)}

The bare two-photon f02/2 feature lies at {config.anharmonicity_mhz/2:.4f} MHz,
inside the requested span. It is not f01; at high amplitude its position and line shape also change.

## Validation and files

Full Lindblad matrix exponentials on {metadata['segments_per_half']} midpoint segments per half.
Quadrupling the segment count on 27 amplitude/detuning probes changed final
populations by at most {validation['max_population_change_quadrupled_segments']:.3g}.
Adding a fifth level changed probe P1 by at most
{validation['max_P1_change_four_to_five_levels_on_probes']:.3g}; the 100 MHz dressed
resonance changed by {validation['f01_center_change_four_to_five_levels_at_100mhz']:.6g} MHz.
These are sampled convergence checks, not bounds over the whole map.

- spectroscopy.npz: both protocols, all four populations, broad and fine frequency grids.
- resonance_vs_amplitude.csv / resonance_analysis.npz: analytical, dressed and operational centers.
- refined_peak_spectra.npz / refined_peak_centers.npz: dense local spectra and refined maxima.
- metadata.json: exact config, raw state physicality diagnostics and convergence checks.
- broad_spectroscopy.png / .pdf and state_populations.png: rendered figures.

Reproduce from the repository root:

    .venv/bin/python scripts/make_cutoff_broad_spectroscopy.py --duration-us {config.duration_us:g}

This is a numerical model result, not a hardware measurement.
"""
    (output / "README.md").write_text(report)
    print("100 MHz: dressed", dressed[-1], "quadratic", theory[-1],
          "P1 peaks", centers["ordinary_peak_sigma_0.5"][-1], centers["echo_peak_sigma_0.5"][-1], flush=True)


def render_ordinary_four_level(config, source, data, metadata):
    """Export the already simulated ordinary branch with four separate levels."""
    if config.levels != 4 or data["ordinary_population"].shape[-1] != 4:
        raise ValueError("The ordinary-only report requires exactly four levels")
    output = source / "ordinary_four_level"
    output.mkdir(exist_ok=True)
    a, x, z = (data[k] for k in ("rabi_mhz", "detuning_mhz", "zoom_detuning_mhz"))
    p = data["ordinary_population"]
    zoom = data["ordinary_zoom_population"]
    analysis = dict(np.load(source / "resonance_analysis.npz"))
    np.testing.assert_array_equal(analysis["rabi_mhz"], a)
    analysis = {key: value for key, value in analysis.items() if not key.startswith("echo_")}
    dressed = analysis["dressed_f01_shift_mhz"]
    theory = analysis["quadratic_stark_shift_mhz"]
    peak = analysis["ordinary_peak_sigma_0.5"]
    np.savez_compressed(output / "spectroscopy.npz", rabi_mhz=a, detuning_mhz=x,
                        zoom_detuning_mhz=z, populations=p, zoom_populations=zoom)
    np.savez_compressed(output / "resonance_analysis.npz", **analysis)
    with (output / "resonance_vs_amplitude.csv").open("w") as file:
        writer = csv.writer(file)
        writer.writerow(analysis)
        writer.writerows(zip(*analysis.values(), strict=True))
    refined = dict(np.load(source / "refined_peak_spectra.npz"))
    refined = {key: value for key, value in refined.items() if not key.startswith("echo_")}
    np.savez_compressed(output / "refined_peak_spectra.npz", **refined)
    ordinary_metadata = {
        "config": asdict(replace(config, echo=False)),
        "state_labels": ["|0>", "|1>", "|2>", "|3>"],
        "model": metadata["model"],
        "detuning_convention": metadata["detuning_convention"],
        "segments_per_half": metadata["segments_per_half"],
        "diagnostics": metadata["diagnostics"]["ordinary"],
        "source": str(source / "spectroscopy.npz"),
        "provenance": "Ordinary-pulse branch reused exactly from the existing four-level simulation.",
        "source_validation": metadata["validation"],
        "validation_scope": "Conservative probe maxima from the source run across both pulse protocols.",
        "peak_extraction": "20 kHz local grid; Gaussian sigma 0.5 MHz; sensitivity sigma 0.25–1 MHz",
    }
    (output / "metadata.json").write_text(json.dumps(ordinary_metadata, indent=2) + "\n")

    plt.rcParams.update({"font.size": 10, "axes.spines.top": False,
                         "axes.spines.right": False, "savefig.dpi": 190})
    fig, axes = plt.subplots(2, 2, figsize=(13, 9), layout="constrained")
    ax = axes[0, 0]
    im = ax.pcolormesh(x, a, p[..., 1], shading="auto", cmap="viridis",
                       vmin=0, vmax=0.65, rasterized=True)
    ax.plot(dressed, a, color="white", lw=1.7, label="Dressed f01")
    ax.plot(theory, a, "--", color="#ffb56b", lw=1.5, label="Quadratic theory")
    ax.axvline(0, color="white", ls=":", lw=0.8)
    ax.axvline(config.anharmonicity_mhz / 2, color="white", ls=":", lw=0.8)
    ax.set(xlabel=r"Drive detuning $f_d-f_{01}$ (MHz)", ylabel="Peak Rabi frequency (MHz)",
           title=r"Ordinary pulse: final $P_1$", xlim=(-125, 125), ylim=(0, 100))
    ax.legend(loc="upper left", fontsize=8, facecolor="#243344", labelcolor="white")
    fig.colorbar(im, ax=ax, label=r"$P_1$", fraction=0.045)

    ax = axes[0, 1]
    ax.plot(a, theory, "--", color="#cf652b", lw=2, label="Quadratic AC Stark prediction")
    ax.plot(a, dressed, color="#222222", lw=2.2, label="Dressed f01: four levels")
    ax.plot(a, peak, color="#2076b4", lw=1.5, label="Smoothed P1 response maximum")
    bounds = np.stack([analysis[f"ordinary_peak_sigma_{s:g}"] for s in (0.25, 0.5, 1.0)])
    ax.fill_between(a, np.min(bounds, axis=0), np.max(bounds, axis=0), color="#2076b4", alpha=0.15)
    ax.set(xlabel="Peak Rabi frequency (MHz)", ylabel="Shift from bare f01 (MHz)",
           title="Resonance versus drive amplitude", xlim=(0, 100))
    ax.legend(fontsize=8)
    ax.grid(alpha=0.18)

    ax = axes[1, 0]
    for amp, color in ((20, "#627bbb"), (50, "#18a18d"), (100, "#d98232")):
        i = int(np.argmin(abs(a - amp)))
        ax.plot(z, zoom[i, :, 1], color=color, lw=1, label=f"Rabi = {a[i]:g} MHz")
        ax.axvline(dressed[i], color=color, ls="--", lw=1)
    ax.set(xlabel=r"Drive detuning $f_d-f_{01}$ (MHz)", ylabel=r"Final $P_1$",
           title="Ordinary-pulse spectra; dashed = dressed f01", xlim=(-20, 75))
    ax.legend(fontsize=9)
    ax.grid(alpha=0.18)

    ax = axes[1, 1]
    for level, color in enumerate(("#444444", "#2076b4", "#d98232", "#9b59b6")):
        ax.plot(x, p[-1, :, level], color=color, lw=1.5, label=rf"$P_{level}$")
    ax.axvline(dressed[-1], color="black", ls="--", lw=1)
    ax.set(xlabel=r"Drive detuning $f_d-f_{01}$ (MHz)", ylabel="Final population",
           title="All four levels at 100 MHz Rabi", xlim=(-125, 125))
    ax.legend(ncol=4, fontsize=9)
    ax.grid(alpha=0.18)
    fig.suptitle(f"Four-level system | Ordinary pulse | {config.duration_us:g} µs | cutoff {config.cutoff:g}\n"
                 "250 MHz span · Rabi 0–100 MHz · No midpoint phase flip\n"
                 f"α/2π = {config.anharmonicity_mhz:.3f} MHz; T1 = {config.t1_us:.2f} µs; "
                 f"T2 = {config.t2_us:.2f} µs", fontsize=13)
    for suffix in ("png", "pdf"):
        fig.savefig(output / f"four_level_spectroscopy.{suffix}")
    plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), layout="constrained", sharex=True, sharey=True)
    for level, ax in enumerate(axes.flat):
        im = ax.pcolormesh(x, a, p[..., level], shading="auto", cmap="viridis", vmin=0,
                           rasterized=True)
        ax.plot(dressed, a, color="white", lw=1)
        ax.set(title=rf"$P_{level}$", xlim=(-125, 125), ylim=(0, 100))
        fig.colorbar(im, ax=ax, label="Population", fraction=0.045)
        if level >= 2:
            ax.set_xlabel("Drive detuning (MHz)")
        if level % 2 == 0:
            ax.set_ylabel("Peak Rabi frequency (MHz)")
    fig.suptitle(f"Ordinary pulse · Four separate level populations · {config.duration_us:g} µs\n"
                 "Each panel has its own color scale; white line = dressed f01")
    fig.savefig(output / "four_level_populations.png")
    plt.close(fig)
    (output / "README.md").write_text(f"""# Ordinary pulse, four-level spectroscopy

Duration {config.duration_us:g} µs; cutoff {config.cutoff:g}; peak Rabi 0–100 MHz;
drive detuning −125 to +125 MHz. The modeled states are |0>, |1>, |2>, |3>.
There is no midpoint phase flip, DRAG, or Stark compensation.

The existing ordinary-pulse four-level calculation was reused exactly; no new
dynamics calculation was necessary. metadata.json records the source and config.
The four-level Duffing Lindblad model uses the rotating-wave approximation,
initial ground state, alpha/2pi = {config.anharmonicity_mhz:.9f} MHz,
T1 = {config.t1_us:.9f} µs, and T2 = {config.t2_us:.9f} µs.

At 100 MHz Rabi:

- Dressed f01 shift (minimum dressed-state gap, envelope averaged): +{dressed[-1]:.4f} MHz.
- Quadratic analytical prediction: +{theory[-1]:.4f} MHz.
- Smoothed P1 response maximum: approximately +{peak[-1]:.1f} MHz.

The analytical formula is delta_f = −A²<f²>/(2 alpha_f), where A = Omega0/2pi,
alpha_f = alpha/2pi and <f²> = 0.999333133 for this cutoff. The population maximum
also reflects saturation, leakage, and finite-pulse interference. Its shaded
range varies the Gaussian smoothing sigma from 0.25 to 1 MHz and is not a confidence interval.
The phase flip is absent throughout this result.

The broad grid has 0.5 MHz spacing; the f01 window has 0.1 MHz spacing;
local peak extraction uses 0.02 MHz spacing. The broad plots do not resolve
every coherent fringe. Numerical convergence was checked in the source run;
the raw ordinary states had trace error {ordinary_metadata['diagnostics']['trace_error']:.3g}.
This is a simulation, not a hardware measurement.

Files: four_level_spectroscopy.png/.pdf, four_level_populations.png,
spectroscopy.npz (final axis ordered P0, P1, P2, P3), refined_peak_spectra.npz,
resonance_analysis.npz, resonance_vs_amplitude.csv, and metadata.json.

Regenerate from the saved simulation:

    .venv/bin/python scripts/make_cutoff_broad_spectroscopy.py --duration-us {config.duration_us:g} --plot-only --ordinary-only
""")
    print(output, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--duration-us", type=float, default=60)
    parser.add_argument("--segments-per-half", type=int, default=64)
    parser.add_argument("--plot-only", action="store_true")
    parser.add_argument("--ordinary-only", action="store_true",
                        help="Export an ordinary-pulse report with four separate levels")
    args = parser.parse_args()
    output = ROOT / "outputs" / "cutoff_0p999_broad_spectroscopy" / f"{args.duration_us:g}us"
    output.mkdir(parents=True, exist_ok=True)
    config = SimulationConfig(duration_us=args.duration_us, cutoff=0.999,
                              amplitude_mhz=tuple(np.linspace(0, 100, 51)),
                              detuning_mhz=tuple(np.linspace(-125, 125, 501)))
    if args.plot_only:
        data = dict(np.load(output / "spectroscopy.npz"))
        metadata = json.loads((output / "metadata.json").read_text())
        config = SimulationConfig(**metadata["config"])
    else:
        checks = validate(config, args.segments_per_half)
        print("Convergence:", json.dumps(checks), flush=True)
        data, metadata = calculate(config, output, args.segments_per_half)
        metadata["validation"] = checks
        (output / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    if not (output / "refined_peak_centers.npz").exists() or not args.plot_only:
        refine_centers(config, output, data, metadata["segments_per_half"])
    if not args.ordinary_only or not args.plot_only or not (output / "resonance_analysis.npz").exists():
        render(config, output, data, metadata)
    if args.ordinary_only:
        render_ordinary_four_level(config, output, data, metadata)
    print(output, flush=True)


if __name__ == "__main__":
    main()
