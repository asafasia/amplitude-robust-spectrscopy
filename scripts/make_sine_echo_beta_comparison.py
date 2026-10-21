"""Figure 2 parameter comparison of beta=0,0.5,1 with a 16 ns sine echo.

Run: .venv/bin/python scripts/make_sine_echo_beta_comparison.py
Use --plot-only to rebuild figures from saved arrays.
"""

# ruff: noqa: E402
from __future__ import annotations

import argparse
import csv
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
sys.path.insert(0, str(ROOT))
from echospec.analysis.calibration_gaussian import (
    fit_calibration_dip,
    negative_gaussian,
)
from echospec.analysis.dip_metrics import dip_metrics
from echospec.simulation.qutrit import simulate_qutrit_map, sine_echo_sign
from scripts.make_main_central_spectroscopy_experiment import RUNS, load_measurement
from scripts.make_sine_echo_transition_comparison import COMMON

OUT = ROOT / "outputs/sine_echo_beta_comparison"
BETAS = (0.0, 0.5, 1.0)
RABI_MAX_MHZ = None
CONFIG = {**COMMON, "echo_transition_us": 0.016}
COLORS = ("#2878b5", "#ed8b23", "#2d9852")


def simulate(beta, detuning, rabi, steps=20000):
    result = simulate_qutrit_map(
        **{**CONFIG, "drag_beta": beta},
        detuning_mhz=detuning,
        rabi_mhz=rabi,
        num_steps_per_half=steps,
    )
    return np.array([result.ground, result.excited, result.second_excited])


def run_case(beta):
    data = {}
    for view, points in (("narrow", 201), ("broad", 101)):
        path = OUT / f"beta{beta:g}_{view}.npz"
        measurement = load_measurement(RUNS[view, "echo"])
        detuning = np.linspace(*measurement.detuning_mhz[[0, -1]], points)
        maximum = measurement.rabi_mhz[-1] if RABI_MAX_MHZ is None else RABI_MAX_MHZ
        rabi = np.linspace(measurement.rabi_mhz[0], maximum, 100)
        if path.exists():
            data[view] = dict(np.load(path))
            continue
        chunks = []
        for start in range(0, 100, 25):
            chunks.append(simulate(beta, detuning, rabi[start : start + 25]))
            print(f"beta={beta:g} {view}: {start + 25}/100 amplitudes", flush=True)
        populations = np.concatenate(chunks, axis=1)
        np.savez_compressed(
            path, detuning_mhz=detuning, rabi_mhz=rabi, populations=populations
        )
        data[view] = dict(detuning_mhz=detuning, rabi_mhz=rabi, populations=populations)
    narrow = data["narrow"]
    indices = [7, 46, 99]
    fine = simulate(beta, narrow["detuning_mhz"], narrow["rabi_mhz"][indices], 40000)
    coarse = narrow["populations"][:, indices]
    np.savez_compressed(
        OUT / f"beta{beta:g}_convergence.npz",
        fine=fine,
        coarse=coarse,
        detuning_mhz=narrow["detuning_mhz"],
        rabi_mhz=narrow["rabi_mhz"][indices],
    )
    print(
        f"beta={beta:g} convergence max population error {np.max(abs(fine - coarse)):.3g}",
        flush=True,
    )


def extract(detuning, trace):
    fit = fit_calibration_dip(detuning * 1e6, trace)
    # Exclude unresolved, clipped, weak or clearly non-Gaussian dips.
    fit["accepted"] = bool(
        fit["accepted"]
        and fit["r_squared"] >= 0.95
        and fit["contrast"] >= 0.001
        and 15000 <= fit["fwhm_hz"] <= 600000
        and abs(fit["center_hz"]) + fit["fwhm_hz"] / 2 < 300000
    )
    return fit


def savefig(fig, name):
    fig.savefig(OUT / f"{name}.png", dpi=170)
    fig.savefig(OUT / f"{name}.pdf")
    plt.close(fig)


def plots():
    datasets = {
        view: [dict(np.load(OUT / f"beta{b:g}_{view}.npz")) for b in BETAS]
        for view in ("narrow", "broad")
    }
    direct = [
        [dip_metrics(d["detuning_mhz"], trace) for trace in d["populations"][1]]
        for d in datasets["narrow"]
    ]
    direct_rows = [
        dict(beta=beta, rabi_mhz=float(amp), **metric)
        for beta, d, metrics in zip(BETAS, datasets["narrow"], direct, strict=True)
        for amp, metric in zip(d["rabi_mhz"], metrics, strict=True)
    ]
    with (OUT / "direct_resonance_metrics.csv").open("w") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(direct_rows[0]))
        writer.writeheader()
        writer.writerows(direct_rows)
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), layout="constrained")
    amps = datasets["narrow"][0]["rabi_mhz"]
    for index, (beta, color, metrics) in enumerate(
        zip(BETAS, COLORS, direct, strict=True)
    ):
        for col, key in enumerate(("center_hz", "fwhm_hz")):
            values = np.array([m[key] for m in metrics]) / 1000
            ref = np.array([m[key] for m in direct[0]]) / 1000
            axes[0, col].plot(amps, values, color=color, label=f"β={beta:g}")
            if index:
                axes[1, col].plot(
                    amps, values - ref, color=color, label=f"β={beta:g} − β=0"
                )
    for ax in axes.flat:
        ax.set_xlabel("Rabi amplitude (MHz)")
        ax.grid(alpha=0.2)
        ax.legend()
    axes[0, 0].set(
        title="Direct Pe minimum (quadratic interpolation)",
        ylabel="Center detuning (kHz)",
    )
    axes[0, 1].set(
        title="Half-depth width after linear background removal", ylabel="FWHM (kHz)"
    )
    axes[1, 0].set(title="Center change relative to β=0", ylabel="Δ center (kHz)")
    axes[1, 1].set(title="Linewidth change relative to β=0", ylabel="Δ FWHM (kHz)")
    savefig(fig, "direct_resonance_and_linewidth")
    rows = []
    fits = []
    for beta, d in zip(BETAS, datasets["narrow"], strict=True):
        beta_fits = [extract(d["detuning_mhz"], trace) for trace in d["populations"][1]]
        fits.append(beta_fits)
        for rabi, fit in zip(d["rabi_mhz"], beta_fits, strict=True):
            rows.append(dict(beta=beta, rabi_mhz=float(rabi), **fit))
    with (OUT / "resonance_fits.csv").open("w") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    for view, data in datasets.items():
        fig, axes = plt.subplots(3, 3, figsize=(13, 10), layout="constrained")
        for state, name in enumerate(("Pg", "Pe", "Pf")):
            vmax = 1 if state < 2 else max(d["populations"][state].max() for d in data)
            for col, (beta, d) in enumerate(zip(BETAS, data, strict=True)):
                ax = axes[state, col]
                im = ax.pcolormesh(
                    d["detuning_mhz"],
                    d["rabi_mhz"],
                    d["populations"][state],
                    vmin=0,
                    vmax=vmax,
                    cmap="viridis",
                    shading="auto",
                    rasterized=True,
                )
                ax.set(
                    title=f"{name} · β = {beta:g}",
                    xlabel="Detuning (MHz)",
                    ylabel="Rabi amplitude (MHz)",
                )
                if view == "narrow" and state == 1:
                    center = np.array(
                        [
                            f["center_hz"] / 1e6 if f["accepted"] else np.nan
                            for f in direct[col]
                        ]
                    )
                    ax.plot(
                        center,
                        d["rabi_mhz"],
                        color="white",
                        lw=1,
                        label="Pe minimum",
                    )
                    ax.legend(fontsize=8)
            fig.colorbar(im, ax=axes[state, :], label=name)
        fig.suptitle(
            f"{view.capitalize()} echo spectroscopy · 16 ns sine transition · 100 amplitudes\n"
            f"20 µs · cutoff {CONFIG['cutoff']:g} · no Stark compensation · final populations"
        )
        savefig(fig, view + "_populations")
    fig, axes = plt.subplots(2, 3, figsize=(14, 8), layout="constrained")
    for col, (beta, color, d) in enumerate(
        zip(BETAS, COLORS, datasets["narrow"], strict=True)
    ):
        rabi = d["rabi_mhz"]
        ff = fits[col]

        def values(key, fit_rows=ff):
            return np.array([f[key] if f["accepted"] else np.nan for f in fit_rows])

        axes[0, 0].plot(
            rabi, values("center_hz") / 1000, color=color, label=f"β={beta:g}"
        )
        axes[0, 1].plot(rabi, values("fwhm_hz") / 1000, color=color)
        axes[0, 2].plot(rabi, np.max(d["populations"][2], axis=1), color=color)
        axes[1, 0].plot(rabi, values("contrast"), color=color)
        axes[1, 1].plot(rabi, [f["r_squared"] for f in ff], color=color)
        axes[1, 2].plot(
            d["detuning_mhz"] * 1000,
            d["populations"][1, -1],
            color=color,
            label=f"β={beta:g}",
        )
        f = ff[-1]
        if f["accepted"]:
            x = np.linspace(-0.3, 0.3, 500)
            # Recover offset from the fitted depth and smoothed central fit.
            from scipy.ndimage import gaussian_filter1d

            y = gaussian_filter1d(d["populations"][1, -1, 40:-40], 1)
            g = negative_gaussian(
                d["detuning_mhz"][40:-40] * 1e6,
                0,
                f["contrast"],
                f["center_hz"],
                f["fwhm_hz"] / 2.354820045,
            )
            offset = np.mean(y - g)
            axes[1, 2].plot(
                x * 1000,
                negative_gaussian(
                    x * 1e6,
                    offset,
                    f["contrast"],
                    f["center_hz"],
                    f["fwhm_hz"] / 2.354820045,
                ),
                "--",
                color=color,
                alpha=0.7,
            )
    axes[0, 0].axhline(0, color="gray", lw=0.6)
    axes[1, 1].axhline(0.95, color="gray", ls="--", lw=0.8)
    labels = [
        ("Gaussian dip center", "Rabi amplitude (MHz)", "Center detuning (kHz)"),
        ("Gaussian dip FWHM", "Rabi amplitude (MHz)", "FWHM (kHz)"),
        ("Maximum final leakage over ±0.5 MHz", "Rabi amplitude (MHz)", "max Pf"),
        ("Fitted dip depth", "Rabi amplitude (MHz)", "Population contrast"),
        (
            "Fit quality; dashed line = acceptance threshold",
            "Rabi amplitude (MHz)",
            "R²",
        ),
        (
            f"{datasets['narrow'][0]['rabi_mhz'][-1]:.2f} MHz drive: data and dashed Gaussian fits",
            "Detuning (kHz)",
            "Pe",
        ),
    ]
    for ax, (title, xlabel, ylabel) in zip(axes.flat, labels, strict=True):
        ax.set(title=title, xlabel=xlabel, ylabel=ylabel)
        ax.grid(alpha=0.2)
    axes[0, 0].legend()
    axes[1, 2].legend()
    savefig(fig, "resonance_and_linewidth")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), layout="constrained")
    rabi = datasets["narrow"][0]["rabi_mhz"]
    for index in (1, 2):
        valid = np.array(
            [
                a["accepted"] and b["accepted"]
                for a, b in zip(fits[index], fits[0], strict=True)
            ]
        )
        for ax, key in zip(axes, ("center_hz", "fwhm_hz"), strict=True):
            delta = (
                np.array(
                    [a[key] - b[key] for a, b in zip(fits[index], fits[0], strict=True)]
                )
                / 1000
            )
            ax.plot(
                rabi,
                np.where(valid, delta, np.nan),
                color=COLORS[index],
                label=f"β={BETAS[index]:g} minus β=0",
            )
            ax.axhline(0, color="gray", lw=0.7)
            ax.grid(alpha=0.2)
            ax.legend()
            ax.set_xlabel("Rabi amplitude (MHz)")
    axes[0].set(title="Resonance center change", ylabel="Δ center (kHz)")
    axes[1].set(title="Linewidth change", ylabel="Δ FWHM (kHz)")
    savefig(fig, "changes_from_beta_zero")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), layout="constrained")
    time = np.linspace(-30, 30, 1201) / 1000
    sigma = (CONFIG["duration_us"] / 2) / np.sqrt(
        CONFIG["cutoff"] ** (-1 / CONFIG["order"]) - 1
    )
    base = (1 + (time / sigma) ** 2) ** -0.5
    derivative = -time / sigma**2 * (1 + (time / sigma) ** 2) ** -1.5
    signs, slopes = np.array([sine_echo_sign(t, 0.016) for t in time]).T
    axes[0].plot(time * 1000, base * signs)
    for beta, color in zip(BETAS, COLORS, strict=True):
        q = -beta * (derivative * signs + base * slopes) / (2 * np.pi * 216)
        axes[1].plot(time * 1000, q, color=color, label=f"β={beta:g}")
    axes[0].set(
        title="I: identical for all beta",
        xlabel="Time from midpoint (ns)",
        ylabel="I / original peak",
    )
    axes[1].set(
        title="Q = −β dI/dt / (2π |α|)",
        xlabel="Time from midpoint (ns)",
        ylabel="Q / original peak",
    )
    axes[1].legend()
    savefig(fig, "waveforms")
    summary = []
    for beta, d, ff in zip(BETAS, datasets["narrow"], fits, strict=True):
        check = dict(np.load(OUT / f"beta{beta:g}_convergence.npz"))
        center_errors, width_errors = [], []
        direct_center_errors, direct_width_errors = [], []
        for coarse, fine in zip(check["coarse"][1], check["fine"][1], strict=True):
            da = dip_metrics(check["detuning_mhz"], coarse)
            db = dip_metrics(check["detuning_mhz"], fine)
            if da["accepted"] and db["accepted"]:
                direct_center_errors.append(abs(da["center_hz"] - db["center_hz"]))
                direct_width_errors.append(abs(da["fwhm_hz"] - db["fwhm_hz"]))
            a, b = (
                extract(check["detuning_mhz"], coarse),
                extract(check["detuning_mhz"], fine),
            )
            if a["accepted"] and b["accepted"]:
                center_errors.append(abs(a["center_hz"] - b["center_hz"]))
                width_errors.append(abs(a["fwhm_hz"] - b["fwhm_hz"]))
        summary.append(
            dict(
                beta=beta,
                max_final_pf=float(d["populations"][2].max()),
                accepted_fits=sum(f["accepted"] for f in ff),
                high_drive_fit=ff[-1],
                high_drive_direct=dip_metrics(
                    d["detuning_mhz"], d["populations"][1, -1]
                ),
                max_population_error=float(
                    np.max(abs(check["fine"] - check["coarse"]))
                ),
                max_checked_direct_center_error_hz=max(
                    direct_center_errors, default=None
                ),
                max_checked_direct_fwhm_error_hz=max(direct_width_errors, default=None),
                max_checked_center_error_hz=max(center_errors, default=None),
                max_checked_fwhm_error_hz=max(width_errors, default=None),
            )
        )
    provenance = dict(
        parameters=CONFIG,
        betas=BETAS,
        amplitude_points=100,
        amplitude_range_mhz=[
            float(datasets["narrow"][0]["rabi_mhz"][0]),
            float(datasets["narrow"][0]["rabi_mhz"][-1]),
        ],
        dt_ns=0.5,
        summary=summary,
        source_runs={v: RUNS[v, "echo"] for v in ("broad", "narrow")},
        convention="I=L*s, no renormalization; Q=-beta*dI/dt/(2*pi*abs(alpha)); drive=I-iQ",
        direct_metrics="Center: raw Pe minimum refined quadratically. Width: nearest half-depth crossings after subtracting linear background fitted on |detuning|>=0.4 MHz; finite-window operational width.",
        fitting="Pe Gaussian dip on ±0.3 MHz after 1-bin Gaussian smoothing; FWHM=2sqrt(2ln2)*sigma. Require R2>=0.95, depth>=0.001, FWHM>=15 kHz, and both half-depth points inside fit window. Zero drive excluded. No uncertainty from fit covariance is interpreted as physical uncertainty.",
        caveat="Final Pf, not transient peak leakage; three-level Markovian simulation. Fitted center and FWHM describe the Gaussian approximation, not necessarily the exact minimum or literal half-depth width.",
    )
    (OUT / "provenance.json").write_text(json.dumps(provenance, indent=2))
    body = """<!doctype html><meta charset='utf-8'><title>16 ns echo: DRAG beta comparison</title>
    <style>body{font:17px system-ui;max-width:1450px;margin:30px auto;padding:20px}img{width:100%}pre{white-space:pre-wrap}td,th{padding:10px;text-align:right}</style>
    <h1>16 ns sine echo: β = 0, 0.5, 1</h1>
    <p>100 amplitude points over the original Figure 2 range (0–42.7 MHz). 20 µs pulse, cutoff 0.005,
    T1=51.24 µs, T2=7.31 µs, α=−216 MHz. Identical I envelopes; no Stark compensation.
    All results are simulated final populations. White map curves show directly located Pe minima.</p>
    <p>The direct comparison uses the raw Pe minimum, refined quadratically, and a half-depth width after subtracting a linear background fitted to |detuning| ≥ 0.4 MHz. This operational width depends on the chosen background. No smoothing is applied.</p>
    <p>The additional Gaussian center and linewidth use a Gaussian fit to Pe over ±0.3 MHz, with one-bin smoothing.
    Poor, weak, unresolved and clipped fits are omitted. The fit-quality plot and example fits expose deviations from a Gaussian shape.</p>
    """
    body += "<h2>At the highest drive, 42.72 MHz</h2><table><tr><th>β</th><th>Pe minimum (kHz)</th><th>Half-depth width (kHz)</th><th>Max final Pf over narrow map (%)</th></tr>"
    for item in summary:
        metric = item["high_drive_direct"]
        body += (
            f"<tr><td>{item['beta']:g}</td><td>{metric['center_hz'] / 1000:.2f}</td>"
            f"<td>{metric['fwhm_hz'] / 1000:.2f}</td><td>{100 * item['max_final_pf']:.5f}</td></tr>"
        )
    body += "</table><p>These high-drive values are one slice of the amplitude-dependent curves below.</p>"
    body += "<p><a href='direct_resonance_metrics.csv'>Direct center/linewidth CSV</a> · <a href='resonance_fits.csv'>Gaussian fits CSV</a> · <a href='provenance.json'>Parameters and convergence</a></p>"
    for name in (
        "narrow_populations",
        "direct_resonance_and_linewidth",
        "resonance_and_linewidth",
        "changes_from_beta_zero",
        "broad_populations",
        "waveforms",
    ):
        body += f"<img src='{name}.png' alt='{name}'>"
    body += (
        "<h2>Numerical convergence and parameters</h2><pre>"
        + json.dumps(provenance, indent=2)
        + "</pre>"
    )
    body = body.replace("β = 0, 0.5, 1", "β = " + ", ".join(f"{b:g}" for b in BETAS))
    body = body.replace("cutoff 0.005", f"cutoff {CONFIG['cutoff']:g}")
    maximum = datasets["narrow"][0]["rabi_mhz"][-1]
    body = body.replace("42.72 MHz", f"{maximum:.2f} MHz")
    if RABI_MAX_MHZ is not None:
        body = body.replace(
            "original Figure 2 range (0–42.7 MHz)",
            f"extended range (0–{maximum:g} MHz)",
        )
    (OUT / "report.html").write_text(body)
    print(json.dumps(summary, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--plot-only", action="store_true")
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if not args.plot_only:
        with ProcessPoolExecutor(max_workers=3) as pool:
            list(pool.map(run_case, BETAS))
    plots()


if __name__ == "__main__":
    main()
