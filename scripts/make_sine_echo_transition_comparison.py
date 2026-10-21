"""Compare Figure 2 echo pulses with finite sine windows at beta=0.

Run from repository root: .venv/bin/python scripts/make_sine_echo_transition_comparison.py
"""

# ruff: noqa: E402
from __future__ import annotations

import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/ars-matplotlib-cache")
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from echospec.simulation.qutrit import simulate_qutrit_map, sine_echo_sign
from scripts.make_main_central_spectroscopy_experiment import RUNS, load_measurement

OUT = ROOT / "outputs/sine_echo_transition_comparison"
WIDTHS = (0, 4, 8, 16)
COMMON = dict(
    duration_us=20.0,
    t1_us=51.24,
    t_phi_us=1 / (1 / 7.31 - 1 / (2 * 51.24)),
    anharmonicity_mhz=-216.0,
    cutoff=0.005,
    order=0.5,
    echo=True,
    echo_transition_shape="sine",
    drag_beta=0.0,
    stark_kappa_mhz_inv=0.0,
)


def run_case(width):
    output = {}
    for view, points in (("broad", 101), ("narrow", 201)):
        measurement = load_measurement(RUNS[view, "echo"])
        detuning = np.linspace(*measurement.detuning_mhz[[0, -1]], points)
        rabi = np.linspace(*measurement.rabi_mhz[[0, -1]], 25)
        result = simulate_qutrit_map(
            **COMMON,
            echo_transition_us=width / 1000,
            detuning_mhz=detuning,
            rabi_mhz=rabi,
            num_steps_per_half=20000,
        )
        output[view + "_populations"] = np.array(
            [result.ground, result.excited, result.second_excited]
        )
        output[view + "_detuning_mhz"] = detuning
        output[view + "_rabi_mhz"] = rabi
        print(f"T={width} ns: {view} complete", flush=True)
    np.savez_compressed(OUT / f"T{width}ns.npz", **output)
    return width


def convergence():
    errors = {}
    for width in WIDTHS:
        populations = []
        for steps in (20000, 40000):
            result = simulate_qutrit_map(
                **COMMON,
                echo_transition_us=width / 1000,
                detuning_mhz=np.array([-25, -0.5, -0.05, 0, 0.05, 0.5, 25]),
                rabi_mhz=np.array([3, 20, 42.72262747]),
                num_steps_per_half=steps,
            )
            populations.append(
                np.array([result.ground, result.excited, result.second_excited])
            )
        errors[str(width)] = float(np.max(abs(populations[1] - populations[0])))
        print(f"Convergence T={width}: {errors[str(width)]:.3g}", flush=True)
    return errors


def plots(errors):
    data = [dict(np.load(OUT / f"T{w}ns.npz")) for w in WIDTHS]
    metrics = []
    for view in ("broad", "narrow"):
        fig, axes = plt.subplots(3, 4, figsize=(15, 9), layout="constrained")
        for row, label in enumerate(("Pg", "Pe", "Pf")):
            vmax = (
                1 if row < 2 else max(d[view + "_populations"][row].max() for d in data)
            )
            for col, (width, d) in enumerate(zip(WIDTHS, data, strict=True)):
                ax = axes[row, col]
                im = ax.pcolormesh(
                    d[view + "_detuning_mhz"],
                    d[view + "_rabi_mhz"],
                    d[view + "_populations"][row],
                    shading="auto",
                    vmin=0,
                    vmax=vmax,
                    cmap="viridis",
                )
                ax.set_title(f"{label} · T = {width} ns")
                if row == 2:
                    ax.set_xlabel("Drive detuning (MHz)")
                if col == 0:
                    ax.set_ylabel("Rabi amplitude (MHz)")
            fig.colorbar(im, ax=axes[row, :], label=label)
        fig.suptitle(
            f"{view.capitalize()} echo spectroscopy · 20 µs · c = 0.005 · β = 0\n"
            "Sine midpoint window; final three-level populations"
        )
        fig.savefig(OUT / f"{view}_populations.png", dpi=160)
        fig.savefig(OUT / f"{view}_populations.pdf")
        plt.close(fig)
    fig, axes = plt.subplots(2, 3, figsize=(14, 8), layout="constrained")
    for width, d in zip(WIDTHS, data, strict=True):
        time = np.linspace(-30, 30, 1201) / 1000
        sigma = 10 / np.sqrt(0.005**-2 - 1)
        env = (1 + (time / sigma) ** 2) ** -0.5
        sign = np.array([sine_echo_sign(t, width / 1000)[0] for t in time])
        axes[0, 0].plot(time * 1000, env * sign, label=f"{width} ns")
        det = d["narrow_detuning_mhz"]
        rabi = d["narrow_rabi_mhz"]
        pop = d["narrow_populations"]
        axes[0, 1].plot(det * 1000, pop[1, -1], label=f"{width} ns")
        axes[0, 2].plot(det * 1000, pop[2, -1])
        axes[1, 0].plot(rabi, pop[1, :, len(det) // 2])
        axes[1, 1].plot(rabi, np.max(pop[2], axis=1))
        axes[1, 2].plot(rabi, np.ptp(pop[1], axis=1))
        metrics.append(
            dict(
                transition_ns=width,
                max_final_pf=float(pop[2].max()),
                high_drive_pe_zero=float(pop[1, -1, len(det) // 2]),
                high_drive_spectral_range=float(np.ptp(pop[1, -1])),
                max_population_change=float(
                    abs(pop - data[0]["narrow_populations"]).max()
                ),
            )
        )
    labels = [
        ("Midpoint waveform (Q = 0)", "Time from midpoint (ns)", "I / original peak"),
        ("Pe at 42.72 MHz drive", "Detuning (kHz)", "Pe"),
        ("Final Pf at 42.72 MHz drive", "Detuning (kHz)", "Pf"),
        ("On-resonance excitation", "Rabi amplitude (MHz)", "Pe at zero detuning"),
        ("Maximum final leakage over ±0.5 MHz", "Rabi amplitude (MHz)", "max Pf"),
        ("Spectral range over ±0.5 MHz", "Rabi amplitude (MHz)", "max Pe − min Pe"),
    ]
    for ax, (title, x, y) in zip(axes.flat, labels, strict=True):
        ax.set(title=title, xlabel=x, ylabel=y)
        ax.grid(alpha=0.2)
    axes[0, 0].legend()
    axes[0, 1].legend()
    fig.savefig(OUT / "diagnostics.png", dpi=160)
    fig.savefig(OUT / "diagnostics.pdf")
    plt.close(fig)
    fig, axes = plt.subplots(3, 3, figsize=(12, 9), layout="constrained")
    for row in range(3):
        diffs = [
            d["narrow_populations"][row] - data[0]["narrow_populations"][row]
            for d in data[1:]
        ]
        limit = max(np.max(abs(a)) for a in diffs)
        for col, delta in enumerate(diffs):
            ax = axes[row, col]
            im = ax.pcolormesh(
                det, rabi, delta, cmap="RdBu_r", vmin=-limit, vmax=limit, shading="auto"
            )
            ax.set(
                title=f"Δ{('Pg', 'Pe', 'Pf')[row]}: {WIDTHS[col + 1]} ns − abrupt",
                xlabel="Detuning (MHz)",
                ylabel="Rabi amplitude (MHz)",
            )
        fig.colorbar(im, ax=axes[row, :])
    fig.savefig(OUT / "population_changes.png", dpi=160)
    plt.close(fig)
    provenance = dict(
        parameters=COMMON,
        transition_ns=WIDTHS,
        dt_ns=0.5,
        convergence_max_abs_population_error=errors,
        metrics=metrics,
        source_runs={v: RUNS[v, "echo"] for v in ("broad", "narrow")},
        grid_note="25 amplitudes; 101 broad / 201 narrow detunings; original Figure 2 endpoints",
        convention="I=L*s without peak renormalization; beta=0, Q=0; abrupt reference separate",
        caveat="Final Pf is not peak transient leakage. Three-level Markovian simulation, not hardware data.",
    )
    (OUT / "provenance.json").write_text(json.dumps(provenance, indent=2))
    body = """<!doctype html><meta charset='utf-8'><title>Sine echo comparison</title>
    <style>body{font:17px system-ui;max-width:1500px;margin:32px auto;padding:20px}img{width:100%}pre{white-space:pre-wrap}</style>
    <h1>Echo midpoint smoothing: 0, 4, 8, 16 ns</h1>
    <p>Figure 2 pulse settings: 20 µs root-Lorentzian, cutoff 0.005, β=0 and Q=0 throughout.
    Sine sign transition over [−T/2,T/2], with no peak renormalization or Stark compensation.
    T1=51.24 µs, T2=7.31 µs, anharmonicity −216 MHz. All populations are simulated final populations.</p>
    """
    for name in (
        "narrow_populations",
        "broad_populations",
        "population_changes",
        "diagnostics",
    ):
        body += f"<img src='{name}.png' alt='{name}'>"
    body += (
        "<h2>Parameters and numerical checks</h2><pre>"
        + json.dumps(provenance, indent=2)
        + "</pre>"
    )
    (OUT / "report.html").write_text(body)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    with ProcessPoolExecutor(max_workers=4) as pool:
        list(pool.map(run_case, WIDTHS))
    errors = convergence()
    plots(errors)


if __name__ == "__main__":
    main()
